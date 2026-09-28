package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.compression.SelectiveContentReducer
import com.siddu.offlinedocai.ai.profiler.DocumentProfile
import com.siddu.offlinedocai.ai.profiler.DocumentProfiler
import com.siddu.offlinedocai.ai.profiler.DocumentSection
import com.siddu.offlinedocai.ai.reformulation.QueryIntent
import com.siddu.offlinedocai.ai.reformulation.QueryReformulator
import com.siddu.offlinedocai.data.entities.ChunkEntity
import org.junit.Assert.*
import org.junit.Test

/**
 * Automated Golden Query Benchmark Test Suite.
 * Validates the 20 golden queries and confirms intent classification, dominance guards,
 * and Selective Content Reduction (SCR) behavior.
 */
class RAGRetrievalBenchmarkTest {

    private val sampleProfile = DocumentProfile(
        schemaVersion = 1,
        documentId = 1L,
        title = "Grafting — A Practical 15-Page Guide",
        domain = "Plant Propagation / Horticulture",
        pageCount = 15,
        sections = listOf(
            DocumentSection(1, 2, "Introduction & Propagation Principles"),
            DocumentSection(3, 4, "Biology Behind a Graft Union"),
            DocumentSection(5, 8, "Grafting Techniques & Plant Selection"),
            DocumentSection(9, 12, "Tools, Timing & Aftercare"),
            DocumentSection(13, 15, "Practical Field Checklist & Troubleshooting")
        ),
        coreGlossary = listOf("scion", "rootstock", "cambium layer", "cleft graft"),
        suggestedQuestions = listOf(
            "What is the basic definition and purpose of grafting?",
            "How does the cambium layer form a graft union?",
            "How do I choose the right plant material for grafting?",
            "What are the most common causes of graft failure?"
        )
    )

    @Test
    fun testGoldenSet_PageNavigation() {
        val navQueries = mapOf(
            "page 5" to 5,
            "show me page 12" to 12,
            "p. 3" to 3,
            "go to page 1" to 1,
            "page 14" to 14
        )
        for ((q, expectedPage) in navQueries) {
            val res = QueryReformulator.classifyAndRewrite(q, sampleProfile)
            assertTrue("Expected PageNavigation for '$q'", res.intent is QueryIntent.PageNavigation)
            assertEquals("Expected target page $expectedPage for '$q'", expectedPage, (res.intent as QueryIntent.PageNavigation).targetPage)
        }
    }

    @Test
    fun testGoldenSet_Definitional() {
        val defQueries = listOf(
            "what is grafting",
            "what is rootstock",
            "what is cambium layer",
            "scion meaning",
            "tell me about graft union"
        )
        for (q in defQueries) {
            val res = QueryReformulator.classifyAndRewrite(q, sampleProfile)
            assertTrue("Expected Definitional for '$q' but was ${res.intent}", res.intent is QueryIntent.Definitional)
            assertTrue("Rewritten query should contain expansion tokens", res.searchString.contains("definition"))
        }
    }

    @Test
    fun testGoldenSet_Procedural() {
        val procQueries = listOf(
            "how to choose rootstock",
            "how to select plant for grafting",
            "how to do whip and tongue graft",
            "troubleshooting graft failure",
            "when should i graft",
            "tools needed for grafting"
        )
        for (q in procQueries) {
            val res = QueryReformulator.classifyAndRewrite(q, sampleProfile)
            assertTrue("Expected Procedural for '$q' but was ${res.intent}", res.intent is QueryIntent.Procedural)
            assertTrue("Rewritten query should contain procedure anchors", res.searchString.contains("procedure") || res.searchString.contains("guide"))
        }
    }

    @Test
    fun testGoldenSet_GlobalSummary() {
        val summaryQueries = listOf(
            "summarize this document",
            "what is this doc about",
            "TL;DR",
            "main points"
        )
        for (q in summaryQueries) {
            val res = QueryReformulator.classifyAndRewrite(q, sampleProfile)
            assertTrue("Expected GlobalSummary for '$q' but was ${res.intent}", res.intent is QueryIntent.GlobalSummary)
        }
    }

    @Test
    fun testDominanceGuard_PreventsFalseSummary() {
        val specificQueries = listOf(
            "overview of cambium layer",
            "overview of grafting techniques"
        )
        for (q in specificQueries) {
            val res = QueryReformulator.classifyAndRewrite(q, sampleProfile)
            assertFalse("Query '$q' should NOT be classified as GlobalSummary", res.intent is QueryIntent.GlobalSummary)
        }
    }

    @Test
    fun testSectionAffinity() {
        val defIntent = QueryIntent.Definitional("grafting", "grafting definition")
        val introAffinity = QueryReformulator.computeSectionAffinity(defIntent, pageNumber = 1, sampleProfile.sections)
        assertEquals("Definitional intent should match page 1 intro section", 1.0, introAffinity, 0.001)

        val procIntent = QueryIntent.Procedural("whip and tongue", "whip and tongue techniques")
        val procAffinity = QueryReformulator.computeSectionAffinity(procIntent, pageNumber = 6, sampleProfile.sections)
        assertEquals("Procedural intent should match page 6 techniques section", 1.0, procAffinity, 0.001)
    }

    @Test
    fun testSelectiveContentReducer() {
        val longChunk = ChunkEntity(
            id = 1L,
            documentId = 1L,
            pageNumber = 1,
            text = "Grafting is a horticultural technique whereby tissues of plants are joined. " +
                "The upper part of the combined plant is called the scion while the lower part is the rootstock. " +
                "Random unrelated sentence about weather conditions in early spring. " +
                "Another filler sentence containing table footers and copyright disclaimer text 2026."
        )
        val compressed = SelectiveContentReducer.reduceChunks(
            chunks = listOf(longChunk),
            query = "what is grafting",
            retentionRatio = 0.50f
        )
        assertEquals(1, compressed.size)
        assertTrue("Compression should reduce token count", compressed[0].tokenReductionPercentage > 0f)
        assertTrue("Compressed text should retain the core definition", compressed[0].compressedText.contains("horticultural technique"))
    }
}
