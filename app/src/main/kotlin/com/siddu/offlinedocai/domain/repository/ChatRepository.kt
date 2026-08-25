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
        if (documentId != null && documentId > 0) {
            val doc = db.documentDao().getById(documentId)
            val docName = doc?.name ?: "Selected Document"
            val pageCount = doc?.pageCount ?: 0
            
            val welcomeText = """
👋 Welcome to your document assistant! 

I have parsed **$docName** ($pageCount pages), computed semantic embeddings, and successfully indexed the text chunks into local memory database.

### Document Summary:
This document is ready for local Retrieval-Augmented Generation (RAG) queries. You can ask me to search, summarize, or retrieve specific key details from it.

How may I help you with it?
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

    /** Send a user message, run RAG or LLM, and stream the assistant response */
    fun sendMessage(conversationId: Long, userText: String, relevantDocumentIds: List<Long>) {
        CoroutineScope(Dispatchers.IO).launch {
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

                // Build prompt based on whether valid documents are linked
                val prompt = if (cleanDocIds.isEmpty()) {
                    buildNormalPrompt(userText)
                } else {
                    // Embedding of the query
                    val queryEmbedding = embeddingEngine.getEmbedding(userText)

                    // Retrieve relevant chunks (top‑5)
                    val chunks = vectorStore.retrieve(queryEmbedding, cleanDocIds, topK = 5, query = userText)
                    val contextText = chunks.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }

                    // Build prompt
                    buildPrompt(contextText, userText)
                }

                // Generate full answer from GGUFLLM
                val answer = llm.generate(prompt)

                // Insert empty assistant message to initialize streaming updates
                val assistantMsgId = messageDao.insert(
                    MessageEntity(
                        id = 0,
                        conversationId = conversationId,
                        role = "assistant",
                        content = "",
                        timestamp = System.currentTimeMillis()
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
                            timestamp = System.currentTimeMillis()
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
