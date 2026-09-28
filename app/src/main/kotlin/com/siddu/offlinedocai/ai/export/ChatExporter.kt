package com.siddu.offlinedocai.ai.export

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import com.siddu.offlinedocai.data.entities.MessageEntity
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Utility to export the complete conversation history into structured JSON
 * containing user queries, AI responses, and retrieved context chunks.
 * Designed for analysis by AI agents and researchers.
 */
object ChatExporter {

    /**
     * Formats conversation into a pretty-printed JSON string.
     */
    fun buildExportJson(
        conversationId: Long,
        conversationTitle: String,
        documentId: Long?,
        messages: List<MessageEntity>
    ): String {
        val root = JSONObject()
        val isoFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)

        root.put("conversationId", conversationId)
        root.put("conversationTitle", conversationTitle)
        root.put("documentId", documentId ?: -1)
        root.put("exportedAt", isoFormat.format(Date()))
        root.put("totalMessages", messages.size)

        // 1. Structured Q&A Turns (Ideal for AI agent evaluation)
        val turnsArray = JSONArray()
        var pendingQuestion: MessageEntity? = null

        for (msg in messages) {
            val roleLower = msg.role.lowercase()
            if (roleLower == "user") {
                pendingQuestion = msg
            } else if (roleLower == "assistant" || roleLower == "ai") {
                val turnObj = JSONObject()
                turnObj.put("question", pendingQuestion?.content ?: "(System / Initial Greeting)")
                turnObj.put("questionTimestamp", pendingQuestion?.timestamp ?: 0L)
                turnObj.put("response", msg.content)
                turnObj.put("responseTimestamp", msg.timestamp)

                val chunksArray = JSONArray()
                if (!msg.sources.isNullOrBlank()) {
                    msg.sources.split("|||")
                        .map { it.trim() }
                        .filter { it.isNotEmpty() }
                        .forEach { chunkText ->
                            chunksArray.put(chunkText)
                        }
                }
                turnObj.put("retrievedChunks", chunksArray)
                turnsArray.put(turnObj)
                pendingQuestion = null
            }
        }
        root.put("qaTurns", turnsArray)

        // 2. Sequential raw message log
        val rawMessagesArray = JSONArray()
        for (m in messages) {
            val msgObj = JSONObject()
            msgObj.put("id", m.id)
            msgObj.put("role", m.role)
            msgObj.put("content", m.content)
            msgObj.put("timestamp", m.timestamp)
            if (!m.sources.isNullOrBlank()) {
                val srcArray = JSONArray()
                m.sources.split("|||")
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .forEach { srcArray.put(it) }
                msgObj.put("sources", srcArray)
            }
            rawMessagesArray.put(msgObj)
        }
        root.put("rawMessages", rawMessagesArray)

        return root.toString(2)
    }

    /**
     * Copies the formatted JSON export to the system clipboard and displays a Toast confirmation.
     */
    fun copyChatToClipboard(
        context: Context,
        conversationId: Long,
        conversationTitle: String,
        documentId: Long?,
        messages: List<MessageEntity>
    ) {
        if (messages.isEmpty()) {
            Toast.makeText(context, "No chat messages to export", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val jsonString = buildExportJson(conversationId, conversationTitle, documentId, messages)
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("DocAI Chat Export", jsonString)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(context, "Chat exported to clipboard (JSON)!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Failed to export chat: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}
