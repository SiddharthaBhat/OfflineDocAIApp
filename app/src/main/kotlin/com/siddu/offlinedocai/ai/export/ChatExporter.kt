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
     * @param includeChunks When true, includes retrieved context chunks and source citations.
     *                      When false, exports only questions and responses without chunk payloads.
     */
    fun buildExportJson(
        conversationId: Long,
        conversationTitle: String,
        documentId: Long?,
        messages: List<MessageEntity>,
        includeChunks: Boolean = true
    ): String {
        val root = JSONObject()
        val isoFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)

        root.put("conversationId", conversationId)
        root.put("conversationTitle", conversationTitle)
        root.put("documentId", documentId ?: -1)
        root.put("exportedAt", isoFormat.format(Date()))
        root.put("totalMessages", messages.size)
        root.put("includeChunks", includeChunks)

        // 1. Structured Q&A Turns
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

                if (includeChunks) {
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
                }
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
            if (includeChunks && !m.sources.isNullOrBlank()) {
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
     * Formats conversation into clean human-readable Q&A Markdown text without retrieved chunks.
     */
    fun buildPlainTextQnA(
        conversationTitle: String,
        messages: List<MessageEntity>
    ): String {
        val sb = StringBuilder()
        sb.appendLine("# 💬 $conversationTitle")
        sb.appendLine("─".repeat(40))
        sb.appendLine()

        var turnNumber = 1
        var pendingUserText: String? = null

        for (msg in messages) {
            val roleLower = msg.role.lowercase()
            if (roleLower == "user") {
                pendingUserText = msg.content
            } else if (roleLower == "assistant" || roleLower == "ai") {
                if (pendingUserText != null) {
                    sb.appendLine("### Q$turnNumber: $pendingUserText")
                    sb.appendLine()
                    sb.appendLine("**Answer:**")
                    sb.appendLine(msg.content)
                    sb.appendLine()
                    sb.appendLine("─".repeat(40))
                    sb.appendLine()
                    turnNumber++
                    pendingUserText = null
                } else if (msg.content.isNotBlank()) {
                    sb.appendLine("**Greeting / System:**")
                    sb.appendLine(msg.content)
                    sb.appendLine()
                    sb.appendLine("─".repeat(40))
                    sb.appendLine()
                }
            }
        }

        if (pendingUserText != null) {
            sb.appendLine("### Q$turnNumber: $pendingUserText")
            sb.appendLine()
        }

        return sb.toString().trim()
    }

    /**
     * Copies the formatted JSON export to the system clipboard and displays a Toast confirmation.
     * @param includeChunks When true, copies full JSON with chunks; when false, copies Q&A JSON without chunks.
     */
    fun copyChatToClipboard(
        context: Context,
        conversationId: Long,
        conversationTitle: String,
        documentId: Long?,
        messages: List<MessageEntity>,
        includeChunks: Boolean = true
    ) {
        if (messages.isEmpty()) {
            Toast.makeText(context, "No chat messages to copy", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val jsonString = buildExportJson(conversationId, conversationTitle, documentId, messages, includeChunks)
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val label = if (includeChunks) "DocAI Chat Full Export" else "DocAI Chat Q&A Export"
            val clip = ClipData.newPlainText(label, jsonString)
            clipboard.setPrimaryClip(clip)
            val toastMsg = if (includeChunks) {
                "Copied full chat (with chunks) to clipboard!"
            } else {
                "Copied Q&A only (without chunks) to clipboard!"
            }
            Toast.makeText(context, toastMsg, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Failed to copy chat: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Copies clean, human-readable Q&A text to clipboard without JSON syntax or retrieved chunks.
     */
    fun copyPlainTextQnAToClipboard(
        context: Context,
        conversationTitle: String,
        messages: List<MessageEntity>
    ) {
        if (messages.isEmpty()) {
            Toast.makeText(context, "No chat messages to copy", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val plainText = buildPlainTextQnA(conversationTitle, messages)
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("DocAI Chat Q&A Text", plainText)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(context, "Copied Q&A text to clipboard!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Failed to copy text: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}
