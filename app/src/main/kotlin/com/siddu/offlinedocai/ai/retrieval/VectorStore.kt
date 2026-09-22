package com.siddu.offlinedocai.ai.retrieval

import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.FloatArrayConverter
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Enhanced vector and lexical search over retrieved document chunks.
 */
class VectorStore(private val chunkDao: ChunkDao) {

    private val genericQueryKeywords = setOf(
        "about", "summary", "summarize", "overview", "tell", "explain", "main", "document",
        "file", "content", "contents", "topic", "purpose", "describe", "what", "what's"
    )

    suspend fun retrieve(
        queryEmbedding: FloatArray,
        relevantDocumentIds: List<Long>,
        topK: Int,
        query: String = ""
    ): List<ChunkEntity> {
        if (relevantDocumentIds.isEmpty()) return emptyList()

        val chunks = chunkDao.getAllChunksForDocumentIds(relevantDocumentIds)
        if (chunks.isEmpty()) return emptyList()

        val cleanQuery = query.lowercase().trim()
        val queryTokens = cleanQuery.split(Regex("[^a-zA-Z0-9]+"))
            .filter { it.length > 1 }

        val stopWords = setOf(
            "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
            "them", "with", "from", "the", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "a", "an",
            "of", "for", "by", "about", "give", "tell", "explain", "show", "me", "some", "can", "you", "does", "do", "define"
        )
        val keywords = queryTokens.filter { it !in stopWords }

        // Only classify as generic summary query if user explicitly asks for document-wide summary
        val isExplicitSummaryRequest = listOf("summarize document", "summary of document", "overview of document", "document summary").any { cleanQuery.contains(it) }

        if (isExplicitSummaryRequest) {
            if (chunks.size <= topK) return chunks
            val step = chunks.size.toDouble() / topK
            val selected = mutableListOf<ChunkEntity>()
            for (i in 0 until topK) {
                val idx = (i * step).toInt().coerceIn(0, chunks.size - 1)
                if (!selected.contains(chunks[idx])) {
                    selected.add(chunks[idx])
                }
            }
            return selected
        }

        // Dense score evaluation
        var scoredChunks = chunks.mapNotNull { chunk ->
            val embeddingBytes = chunk.embedding ?: return@mapNotNull null
            val chunkEmbedding = FloatArrayConverter.toFloatArray(embeddingBytes) ?: return@mapNotNull null
            val score = cosineSimilarity(queryEmbedding, chunkEmbedding)
            Pair(chunk, score)
        }

        val allZeros = scoredChunks.all { it.second == 0f }

        // Dense score evaluation or BM25/TF-IDF Lexical Fallback
        if (allZeros && query.isNotEmpty()) {
            val docCount = chunks.size.toDouble()
            val idfMap = keywords.associateWith { kw ->
                val containingDocs = chunks.count { it.text.lowercase().contains(kw) }
                if (containingDocs > 0) ln((docCount + 1.0) / (containingDocs + 0.5)) + 1.0 else 0.0
            }

            // Phrase extraction
            val phrase = keywords.joinToString(" ")

            scoredChunks = chunks.map { chunk ->
                val chunkTextLower = chunk.text.lowercase()
                var score = 0.0
                
                // Huge bonus for exact continuous phrase match (e.g. "information security")
                if (phrase.length >= 4 && chunkTextLower.contains(phrase)) {
                    score += 50.0
                }

                // Check for individual keyword matches
                var matchedKeywordCount = 0
                for (kw in keywords) {
                    if (chunkTextLower.contains(kw)) {
                        matchedKeywordCount++
                        val occurrences = chunkTextLower.split(kw).size - 1
                        val tf = occurrences.toDouble() / (chunkTextLower.split(Regex("\\s+")).size.toDouble() + 1.0)
                        val idf = idfMap[kw] ?: 1.0
                        score += tf * idf * 10.0
                    }
                }

                // Extra multiplier if ALL query keywords exist in the chunk
                if (keywords.isNotEmpty() && matchedKeywordCount == keywords.size) {
                    score += 20.0
                }

                Pair(chunk, score.toFloat())
            }.filter { it.second > 0f }
        }

        val sorted = scoredChunks
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }

        return if (sorted.isNotEmpty()) sorted else chunks.take(topK)
    }

    private fun cosineSimilarity(vectorA: FloatArray, vectorB: FloatArray): Float {
        if (vectorA.size != vectorB.size) return 0f
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in vectorA.indices) {
            val a = vectorA[i].toDouble()
            val b = vectorB[i].toDouble()
            dotProduct += a * b
            normA += a * a
            normB += b * b
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom == 0.0) 0f else (dotProduct / denom).toFloat()
    }
}

