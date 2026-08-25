package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * A placeholder implementation for a GGUF language model.
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
            // RAG Response Generation: Extract keywords from question
            val keywords = question.lowercase()
                .split(Regex("[^a-zA-Z0-9]+"))
                .filter { it.length > 3 && it !in listOf("what", "when", "where", "how", "this", "that", "there", "their", "them", "with", "from", "your", "about") }
            
            // Extract sentences from context
            val sentences = contextText.split(Regex("[.!?\n]+"))
                .map { it.trim() }
                .filter { it.isNotEmpty() }
            
            // Score sentences based on keyword matches
            val scoredSentences = sentences.map { sentence ->
                val score = keywords.count { keyword -> sentence.lowercase().contains(keyword) }
                sentence to score
            }.filter { it.second > 0 }
            .sortedByDescending { it.second }
            
            if (scoredSentences.isNotEmpty()) {
                val bestSentences = scoredSentences.map { it.first }
                val sourcePages = Regex("\\[Page (\\d+)\\]").findAll(contextText)
                    .map { it.groupValues[1] }
                    .distinct()
                    .toList()
                
                val sourceInfo = if (sourcePages.isNotEmpty()) " (Source: Page ${sourcePages.joinToString(", ")})" else ""
                
                "Based on the referenced documents, here is what I found:\n\n" +
                bestSentences.joinToString(" ") + sourceInfo
            } else {
                "The document contains information regarding your query, but I couldn't find a direct keyword match. Here is the relevant snippet:\n\n" +
                contextText.take(300) + "..."
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
Here is a detailed explanation regarding $topicName:

$topicName represents a fundamental concept that has wide-ranging implications and practical applications in modern domains. In general, it provides a solid baseline for structuring workflows, developing models, and improving system efficiency.

Key parameters and considerations include:
1. **Structural Design:** Optimizing core nodes and processes to scale dynamically while minimizing processing overhead.
2. **Implementation Strategy:** Utilizing best-practice methodologies to ensure reliability, security, and consistent output results.
3. **Optimizations:** Fine-tuning critical configurations and managing memory limits effectively to ensure robust long-term operation.

Please let me know if you would like me to expand on any specific sub-topic or provide a structured example!
""".trimIndent()
                }
            }
        }
    }
}
