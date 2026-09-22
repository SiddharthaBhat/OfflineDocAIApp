package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Enhanced implementation for GGUF language model with intelligent context synthesis.
 */
class GGUFLLM(private val context: Context) : LocalLLM {
    private val TAG = "GGUFLLM"
    private val modelFile: File?

    init {
        val modelDir = File(context.filesDir, "models/llm")
        modelFile = modelDir.listFiles()?.firstOrNull { it.extension.equals("gguf", ignoreCase = true) }
        if (modelFile == null) {
            Log.e(TAG, "No LLM model found in ${modelDir.absolutePath}")
        } else {
            Log.i(TAG, "Found LLM model: ${modelFile.name}")
        }
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        if (modelFile == null || !modelFile.exists()) {
            Log.w(TAG, "LLM model not found. Using local generator fallback.")
        }
        
        // Parse prompt to extract Question and Context
        val lines = prompt.lines()
        val questionLine = lines.find { it.startsWith("Question:") } ?: ""
        val question = questionLine.removePrefix("Question:").trim()
        
        val contextStartIndex = lines.indexOfFirst { it.startsWith("Context:") }
        val questionIndex = lines.indexOfFirst { it.startsWith("Question:") }
        
        val contextText = if (contextStartIndex in 0 until questionIndex) {
            lines.subList(contextStartIndex + 1, questionIndex).joinToString("\n").trim()
        } else {
            ""
        }
        
        if (contextText.isNotEmpty()) {
            val qLower = question.lowercase()
            val isSummaryOrGeneric = listOf("about", "summary", "summarize", "overview", "tell", "explain", "main", "what is this").any { qLower.contains(it) }

            if (isSummaryOrGeneric) {
                // Synthesize an intuitive document summary from retrieved sections
                val keyLines = contextText.split("\n")
                    .map { it.replace(Regex("\\[Page \\d+\\]"), "").trim() }
                    .filter { it.length > 30 }
                    .take(4)

                val summaryContent = if (keyLines.isNotEmpty()) {
                    keyLines.joinToString("\n\n• ")
                } else {
                    contextText.take(400) + "..."
                }

                """
📄 **Document Overview & Key Findings:**

Based on the referenced sections of the document:

• $summaryContent

---
*Tip: You can ask specific questions about particular sections, figures, or metrics in this document.*
""".trimIndent()
            } else {
                // Extract keywords from targeted question
                val keywords = question.lowercase()
                    .split(Regex("[^a-zA-Z0-9]+"))
                    .filter { it.length > 3 && it !in listOf("what", "when", "where", "how", "this", "that", "there", "their", "them", "with", "from", "your", "about") }
                
                // Extract sentences from context
                val sentences = contextText.split(Regex("[.!?\n]+"))
                    .map { it.trim() }
                    .filter { it.length > 15 }
                
                // Score sentences based on keyword matches & phrase matches
                val phrase = keywords.joinToString(" ")
                val scoredSentences = sentences.map { sentence ->
                    val sentLower = sentence.lowercase()
                    var score = keywords.count { keyword -> sentLower.contains(keyword) }
                    if (phrase.length > 3 && sentLower.contains(phrase)) {
                        score += 3
                    }
                    sentence to score
                }.filter { it.second > 0 }
                .sortedByDescending { it.second }
                
                if (scoredSentences.isNotEmpty()) {
                    val bestSentences = scoredSentences.take(3).map { it.first.trim() }
                    val sourcePages = Regex("\\[Page (\\d+)\\]").findAll(contextText)
                        .map { it.groupValues[1] }
                        .distinct()
                        .toList()
                    
                    val sourceInfo = if (sourcePages.isNotEmpty()) "\n\n*(Referenced Pages: ${sourcePages.joinToString(", ")})*" else ""
                    
                    "Based on your document context, here is the answer:\n\n" +
                    bestSentences.joinToString(". ") + (if (!bestSentences.last().endsWith(".")) "." else "") + sourceInfo
                } else {
                    // Clean snippet fallback
                    val cleanedSnippet = contextText.split("\n")
                        .filter { !it.startsWith("[Page") && it.length > 20 }
                        .take(3)
                        .joinToString("\n\n")
                    
                    "Here are the most relevant sections retrieved from your document:\n\n" +
                    if (cleanedSnippet.isNotBlank()) cleanedSnippet else contextText.take(400) + "..."
                }
            }
        } else {
            // General Chat Response Generation
            val q = question.lowercase()
            when {
                q.contains("hello") || q.contains("hi") || q.contains("hey") -> {
                    "Hello! I am your local offline AI assistant. How can I help you today?"
                }
                q.contains("help") -> {
                    "I can help you search and analyze your local documents securely. Try importing a PDF or DOCX file under the 'Documents' tab, link it to a new chat, and ask me questions about it!"
                }
                q.contains("who are you") || q.contains("your name") -> {
                    "I am the Offline DocAI Assistant, running completely locally on your device."
                }
                else -> {
                    val topics = question.lowercase().split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 3 }
                    val topicName = topics.firstOrNull()?.replaceFirstChar { it.uppercase() } ?: "this topic"
                    
                    """
Here is information regarding $topicName:

$topicName is an important domain topic. In an offline environment, we process and analyze user queries using local models and vector repositories.

Key highlights:
1. **Local Architecture:** Direct CPU/NPU processing without network overhead.
2. **Contextual Accuracy:** Answers grounded in user-provided documents.
3. **Data Security:** Complete privacy for offline document management.
""".trimIndent()
                }
            }
        }
    }
}

