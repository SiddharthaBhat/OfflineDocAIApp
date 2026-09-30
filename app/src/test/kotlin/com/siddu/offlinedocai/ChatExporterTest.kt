package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.export.ChatExporter
import com.siddu.offlinedocai.data.entities.MessageEntity
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ChatExporterTest {

    private val sampleMessages = listOf(
        MessageEntity(
            id = 1L,
            conversationId = 100L,
            role = "assistant",
            content = "Hello! How can I help you?",
            timestamp = 1000L
        ),
        MessageEntity(
            id = 2L,
            conversationId = 100L,
            role = "user",
            content = "What is grafting?",
            timestamp = 2000L
        ),
        MessageEntity(
            id = 3L,
            conversationId = 100L,
            role = "assistant",
            content = "Grafting is a horticultural technique joining plant tissues.",
            timestamp = 3000L,
            sources = "Page 1 [SCR -30%]: Grafting is a horticultural technique...|||Page 2: Scion and rootstock details"
        )
    )

    @Test
    fun testExportWithChunks() {
        val jsonStr = ChatExporter.buildExportJson(
            conversationId = 100L,
            conversationTitle = "Test Chat",
            documentId = 1L,
            messages = sampleMessages,
            includeChunks = true
        )

        val obj = JSONObject(jsonStr)
        assertEquals(100L, obj.getLong("conversationId"))
        assertTrue("includeChunks should be true", obj.getBoolean("includeChunks"))

        val qaTurns = obj.getJSONArray("qaTurns")
        assertEquals(2, qaTurns.length())

        val turn2 = qaTurns.getJSONObject(1)
        assertEquals("What is grafting?", turn2.getString("question"))
        assertTrue("Should have retrieved chunks", turn2.has("retrievedChunks"))
        assertEquals(2, turn2.getJSONArray("retrievedChunks").length())
    }

    @Test
    fun testExportWithoutChunks_QnAOnly() {
        val jsonStr = ChatExporter.buildExportJson(
            conversationId = 100L,
            conversationTitle = "Test Chat",
            documentId = 1L,
            messages = sampleMessages,
            includeChunks = false
        )

        val obj = JSONObject(jsonStr)
        assertFalse("includeChunks should be false", obj.getBoolean("includeChunks"))

        val qaTurns = obj.getJSONArray("qaTurns")
        assertEquals(2, qaTurns.length())

        val turn2 = qaTurns.getJSONObject(1)
        assertEquals("What is grafting?", turn2.getString("question"))
        assertEquals("Grafting is a horticultural technique joining plant tissues.", turn2.getString("response"))
        assertFalse("Should NOT have retrievedChunks in Q&A only mode", turn2.has("retrievedChunks"))

        val rawMessages = obj.getJSONArray("rawMessages")
        for (i in 0 until rawMessages.length()) {
            val msg = rawMessages.getJSONObject(i)
            assertFalse("Raw messages should NOT contain sources in Q&A only mode", msg.has("sources"))
        }
    }

    @Test
    fun testBuildPlainTextQnA() {
        val text = ChatExporter.buildPlainTextQnA(
            conversationTitle = "Grafting Guide",
            messages = sampleMessages
        )

        assertTrue(text.contains("# 💬 Grafting Guide"))
        assertTrue(text.contains("### Q1: What is grafting?"))
        assertTrue(text.contains("**Answer:**"))
        assertTrue(text.contains("Grafting is a horticultural technique joining plant tissues."))
        assertFalse("Plain text should not contain raw chunks markup", text.contains("Page 1 [SCR"))
    }
}
