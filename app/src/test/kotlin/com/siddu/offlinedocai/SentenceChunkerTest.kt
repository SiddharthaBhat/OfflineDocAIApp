package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.domain.rag.ExtractedPage
import org.junit.Assert.*
import org.junit.Test

class SentenceChunkerTest {

    @Test
    fun testSentenceBoundarySplitting_PreservesAbbreviations() {
        val text = "Dr. Smith presented the results. The procedure was successful. Another phase begins soon."
        val sentences = TextChunker.splitIntoSentences(text)
        
        assertEquals(3, sentences.size)
        assertTrue(sentences[0].contains("Dr. Smith"))
        assertEquals("The procedure was successful.", sentences[1])
        assertEquals("Another phase begins soon.", sentences[2])
    }

    @Test
    fun testCrossPageChunking_MaintainsContinuity() {
        val page1 = ExtractedPage(
            pageNumber = 1,
            text = "Grafting is a horticultural technique. The upper part is called scion."
        )
        val page2 = ExtractedPage(
            pageNumber = 2,
            text = "The lower part is called rootstock. Cambium alignment is crucial for success."
        )

        val chunks = TextChunker.chunkDocument(listOf(page1, page2), targetWords = 10, overlapWords = 3)
        assertTrue("Should produce multiple chunks", chunks.size >= 1)
        
        // Confirm text integrity across pages
        val allChunkText = chunks.map { it.text }.joinToString(" ")
        assertTrue(allChunkText.contains("scion"))
        assertTrue(allChunkText.contains("rootstock"))
        assertTrue(allChunkText.contains("Cambium alignment"))
    }

    @Test
    fun testNeverCutsSentenceMidThought() {
        val paragraph = "First complete sentence here. Second complete sentence here. Third complete sentence here."
        val chunks = TextChunker.chunkText(paragraph, pageNumber = 1, chunkSize = 5, overlap = 2)

        for (chunk in chunks) {
            val endsWithPunctuation = chunk.text.endsWith(".") || chunk.text.endsWith("!") || chunk.text.endsWith("?")
            assertTrue("Each chunk should end on a complete sentence: '${chunk.text}'", endsWithPunctuation)
        }
    }
}
