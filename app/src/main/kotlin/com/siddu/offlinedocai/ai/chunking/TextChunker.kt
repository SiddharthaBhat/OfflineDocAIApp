package com.siddu.offlinedocai.ai.chunking

import com.siddu.offlinedocai.domain.rag.ExtractedPage
import java.text.BreakIterator
import java.util.Locale
import kotlin.math.max

/**
 * Intelligent Sentence-Aware & Cross-Page Recursive Document Chunker.
 *
 * Implements:
 * 1. Exact sentence boundary preservation using standard BreakIterator (no mid-sentence truncations).
 * 2. Continuous cross-page span aggregation: sentences crossing page boundaries maintain context.
 * 3. Backward-compatible page-level chunking.
 */
object TextChunker {
    data class Chunk(
        val index: Int,
        val pageNumber: Int,
        val text: String
    )

    private data class PageSentence(
        val text: String,
        val pageNumber: Int,
        val wordCount: Int
    )

    /**
     * Splits text into coherent sentences using BreakIterator.
     * Prevents splits on common abbreviations (e.g., "Dr.", "e.g.", decimal numbers).
     */
    fun splitIntoSentences(text: String): List<String> {
        val clean = text.trim()
        if (clean.isEmpty()) return emptyList()

        val iterator = BreakIterator.getSentenceInstance(Locale.US)
        iterator.setText(clean)

        val rawSentences = mutableListOf<String>()
        var start = iterator.first()
        var end = iterator.next()

        while (end != BreakIterator.DONE) {
            val sentence = clean.substring(start, end).trim()
            if (sentence.isNotEmpty()) {
                rawSentences.add(sentence)
            }
            start = end
            end = iterator.next()
        }

        val abbreviations = setOf("dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "etc.")
        val merged = mutableListOf<String>()
        var i = 0
        while (i < rawSentences.size) {
            var current = rawSentences[i]
            while (i + 1 < rawSentences.size) {
                val lastToken = current.split(Regex("\\s+")).lastOrNull()?.lowercase()?.trimEnd('.', ',') + "."
                if (abbreviations.contains(lastToken)) {
                    i++
                    current = "$current ${rawSentences[i]}"
                } else {
                    break
                }
            }
            merged.add(current)
            i++
        }

        return if (merged.isNotEmpty()) merged else listOf(clean)
    }

    /**
     * Chunks an entire multi-page document with sentence boundary awareness and cross-page continuity.
     *
     * @param pages list of ExtractedPage from DocumentProcessor
     * @param targetWords target word budget per chunk (default 300 words)
     * @param overlapWords word overlap between consecutive chunks (default 60 words)
     */
    fun chunkDocument(
        pages: List<ExtractedPage>,
        targetWords: Int = 300,
        overlapWords: Int = 60
    ): List<Chunk> {
        if (pages.isEmpty()) return emptyList()

        // 1. Extract all sentences with their originating page numbers
        val allSentences = mutableListOf<PageSentence>()
        for (page in pages) {
            val sentences = splitIntoSentences(page.text)
            for (s in sentences) {
                val words = s.split(Regex("\\s+")).filter { it.isNotBlank() }
                if (words.isNotEmpty()) {
                    allSentences.add(PageSentence(s, page.pageNumber, words.size))
                }
            }
        }

        if (allSentences.isEmpty()) return emptyList()

        // 2. Sliding window of whole sentences
        val chunks = mutableListOf<Chunk>()
        var startIndex = 0
        var chunkIndex = 0

        while (startIndex < allSentences.size) {
            var currentWordCount = 0
            var endIndex = startIndex
            val currentSentences = mutableListOf<PageSentence>()

            while (endIndex < allSentences.size) {
                val nextSentence = allSentences[endIndex]
                currentSentences.add(nextSentence)
                currentWordCount += nextSentence.wordCount
                endIndex++

                if (currentWordCount >= targetWords) {
                    break
                }
            }

            if (currentSentences.isNotEmpty()) {
                val chunkText = currentSentences.joinToString(" ") { it.text }
                // Primary page attribution is the page of the central/first substantive sentence
                val primaryPage = currentSentences[currentSentences.size / 2].pageNumber

                chunks.add(
                    Chunk(
                        index = chunkIndex,
                        pageNumber = primaryPage,
                        text = chunkText
                    )
                )
                chunkIndex++
            }

            if (endIndex >= allSentences.size) break

            // Calculate overlap step: retain trailing sentences totaling ~overlapWords
            var overlapCount = 0
            var newStart = endIndex
            for (i in (endIndex - 1) downTo startIndex) {
                overlapCount += allSentences[i].wordCount
                newStart = i
                if (overlapCount >= overlapWords) break
            }

            // Guarantee forward progress
            startIndex = max(startIndex + 1, newStart)
        }

        return chunks
    }

    /**
     * Backward-compatible sentence-aware single-page chunker.
     * Used for single text blocks and tests.
     */
    fun chunkText(
        text: String,
        pageNumber: Int,
        chunkSize: Int = 400,
        overlap: Int = 80
    ): List<Chunk> {
        require(chunkSize > 0) { "chunkSize must be > 0" }
        require(overlap >= 0 && overlap < chunkSize) { "overlap must be >=0 and < chunkSize" }

        val page = ExtractedPage(pageNumber = pageNumber, text = text)
        return chunkDocument(listOf(page), targetWords = chunkSize, overlapWords = overlap)
    }
}
