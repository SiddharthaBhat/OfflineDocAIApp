package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.GGUFEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.LocalLLM
import com.siddu.offlinedocai.ai.llm.OllamaLLM
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.dao.MessageDao
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.util.UUID

/**
 * Repository that implements the RAG chat pipeline.
 */
class ChatRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val vectorStore: VectorStore,
    private val useOllama: Boolean = false // dev flag – set false for final offline mode
) {
    private val TAG = "ChatRepository"
    private val messageDao: MessageDao = db.messageDao()
    private val conversationDao = db.conversationDao()
    private val embeddingEngine: EmbeddingModel = GGUFEmbeddingEngine(context)
    private val llm: LocalLLM = if (useOllama) OllamaLLM() else GGUFLLM(context)

    private val _chatState = MutableStateFlow<ChatState>(ChatState.Idle)
    val chatState: StateFlow<ChatState> = _chatState

    /** Create a new conversation, auto-generate welcome message, and return its ID */
    suspend fun startConversation(title: String = "Chat ${System.currentTimeMillis()}", documentId: Long? = null): Long = kotlinx.coroutines.withContext(Dispatchers.IO) {
        val conversation = ConversationEntity(
            id = 0,
            title = title,
            createdAt = System.currentTimeMillis(),
            documentId = documentId
        )
        val convId = conversationDao.insert(conversation)

        // Insert initial greeting message
        // Insert initial greeting message
        if (documentId != null && documentId > 0) {
            val doc = db.documentDao().getById(documentId)
            val docName = doc?.name ?: "Selected Document"
            val pageCount = doc?.pageCount ?: 0
            val profile = doc?.profileJson?.let { com.siddu.offlinedocai.ai.profiler.DocumentProfiler.fromJson(it) }

            val welcomeText = if (profile != null) {
                val sectionsList = profile.sections.joinToString("\n") { "• **Pages ${it.pageStart}–${it.pageEnd}**: ${it.title}" }
                val conceptsList = profile.coreGlossary.take(5).joinToString(", ")
                val questionsList = profile.suggestedQuestions.joinToString("\n") { "💡 *\"$it\"*" }

                """
📖 **Document Briefing: ${profile.title}**
*Domain: ${profile.domain}* (${profile.pageCount} pages)

### 📌 Summary & Core Concepts:
This guide covers foundational and practical concepts including: **$conceptsList**.

### 🗂️ Document Structure:
$sectionsList

---
### 💡 Suggested Questions:
$questionsList
""".trimIndent()
            } else {
                """
👋 Welcome to your document assistant! 

I have indexed **$docName** ($pageCount pages). You can ask me to search, summarize, or retrieve specific key details from it.

How may I help you today?
""".trimIndent()
            }

            messageDao.insert(
                MessageEntity(
                    id = 0,
                    conversationId = convId,
                    role = "assistant",
                    content = welcomeText,
                    timestamp = System.currentTimeMillis()
                )
            )
        } else {
            val welcomeText = """
👋 Hello! I am your local offline AI assistant. 

How may I help you today?

*Note: This is a general LLM chat session. To query a specific document using RAG, start a new conversation and check the document checkbox to load it as reference.*
""".trimIndent()
            
            messageDao.insert(
                MessageEntity(
                    id = 0,
                    conversationId = convId,
                    role = "assistant",
                    content = welcomeText,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        convId
    }

    /** Process a single message sequentially with RAG/LLM and stream the response */
    suspend fun executeMessage(conversationId: Long, userText: String, relevantDocumentIds: List<Long>) = kotlinx.coroutines.withContext(Dispatchers.IO) {
        try {
            _chatState.value = ChatState.Thinking
            // Persist user message
            val userMsg = MessageEntity(
                id = 0,
                conversationId = conversationId,
                role = "user",
                content = userText,
                timestamp = System.currentTimeMillis()
            )
            messageDao.insert(userMsg)

            // Clean and filter valid document IDs (greater than 0)
            val cleanDocIds = relevantDocumentIds.filter { it > 0 }

            // Check for primary document profile
            val primaryDoc = if (cleanDocIds.isNotEmpty()) db.documentDao().getById(cleanDocIds.first()) else null
            val profile = primaryDoc?.profileJson?.let { com.siddu.offlinedocai.ai.profiler.DocumentProfiler.fromJson(it) }

            // Run Intent Classification and Context-Aware Reformulation
            val reformulation = com.siddu.offlinedocai.ai.reformulation.QueryReformulator.classifyAndRewrite(userText, profile)

            val (prompt, sourcesData, directAnswer) = if (cleanDocIds.isEmpty()) {
                Triple(buildNormalPrompt(userText), null, null)
            } else if (reformulation.intent is com.siddu.offlinedocai.ai.reformulation.QueryIntent.GlobalSummary && profile != null) {
                // Global Summary Short-Circuit (0ms Vector Search)
                val sectionsList = profile.sections.joinToString("\n") { "• **Pages ${it.pageStart}–${it.pageEnd}**: ${it.title}" }
                val summaryAnswer = """
📄 **Executive Summary: ${profile.title}**

This document covers **${profile.domain}** across ${profile.pageCount} pages.

### 🗂️ Document Sections:
$sectionsList

### 📌 Core Key Concepts:
${profile.coreGlossary.joinToString(", ")}

---
*Tip: You can ask specific questions about any of the sections above, or request a specific page (e.g., "page ${profile.sections.firstOrNull()?.pageStart ?: 1}").*
""".trimIndent()
                Triple("", null, summaryAnswer)
            } else {
                // 1. Adaptive Memory Governance (Pocket RAG / HeRo)
                val govConfig = com.siddu.offlinedocai.ai.governor.MemoryGovernor.getAdaptiveConfig()

                // 2. Query Embedding on rewritten search string
                val queryEmbedding = embeddingEngine.getEmbedding(reformulation.searchString)

                // 3. Two-Stage Hybrid Retrieval with Section Prior
                val rawChunks = vectorStore.retrieve(
                    queryEmbedding = queryEmbedding,
                    relevantDocumentIds = cleanDocIds,
                    topK = govConfig.topK,
                    query = reformulation.searchString,
                    intent = reformulation.intent,
                    profile = profile
                )

                // 4. Selective Content Reduction (SCR / SCC)
                val isDefinitional = reformulation.intent is com.siddu.offlinedocai.ai.reformulation.QueryIntent.Definitional
                val compressedChunks = com.siddu.offlinedocai.ai.compression.SelectiveContentReducer.reduceChunks(
                    chunks = rawChunks,
                    query = userText,
                    retentionRatio = if (isDefinitional) 0.85f else govConfig.scrRetentionRatio,
                    isGenericQuery = isDefinitional
                )

                val contextText = com.siddu.offlinedocai.ai.compression.SelectiveContentReducer.formatCompressedContext(compressedChunks)

                // Format chunks as informative source blocks for UI dropdown
                val sourcesJson = compressedChunks.joinToString("|||") {
                    "Page ${it.originalChunk.pageNumber} [SCR -${(it.tokenReductionPercentage * 100).toInt()}%]: ${it.compressedText}"
                }

                // Build prompt
                val p = buildPrompt(contextText, userText)
                Triple(p, sourcesJson, null)
            }

            // Generate full answer from GGUFLLM or direct answer
            val answer = directAnswer ?: llm.generate(prompt)

            // Insert empty assistant message to initialize streaming updates
            val assistantMsgId = messageDao.insert(
                MessageEntity(
                    id = 0,
                    conversationId = conversationId,
                    role = "assistant",
                    content = "",
                    timestamp = System.currentTimeMillis(),
                    sources = sourcesData
                )
            )

            // Simulate token/word streaming updates
            val words = answer.split(" ")
            var currentContent = ""
            for (i in words.indices) {
                currentContent += (if (i > 0) " " else "") + words[i]
                messageDao.update(
                    MessageEntity(
                        id = assistantMsgId,
                        conversationId = conversationId,
                        role = "assistant",
                        content = currentContent,
                        timestamp = System.currentTimeMillis(),
                        sources = sourcesData
                    )
                )
                delay(40) // speed of stream
            }

            _chatState.value = ChatState.Idle
        } catch (e: Exception) {
            Log.e(TAG, "Chat generation failed", e)
            _chatState.value = ChatState.Error(e.message ?: "Unknown error")
        }
    }

    /** Send a user message, run RAG or LLM, and stream the assistant response asynchronously */
    fun sendMessage(conversationId: Long, userText: String, relevantDocumentIds: List<Long>) {
        CoroutineScope(Dispatchers.IO).launch {
            executeMessage(conversationId, userText, relevantDocumentIds)
        }
    }

    private fun buildNormalPrompt(question: String): String {
        return """
You are a helpful offline AI assistant. Answer the user's question directly.

Question: $question

Answer:
""".trimIndent()
    }

    private fun buildPrompt(context: String, question: String): String {
        return """
You are an offline AI assistant. Use the following extracted document context to answer the user's question.

Context:
$context

Question: $question

Answer concisely, referencing the page numbers when appropriate.
""".trimIndent()
    }

    sealed class ChatState {
        object Idle : ChatState()
        object Thinking : ChatState()
        data class Error(val message: String) : ChatState()
    }
}
