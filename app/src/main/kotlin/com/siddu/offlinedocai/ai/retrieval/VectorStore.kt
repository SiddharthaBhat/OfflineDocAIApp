package com.siddu.offlinedocai.ai.retrieval

import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.FloatArrayConverter
import kotlin.math.sqrt

/**
 * Perform vector search on retrieved document chunks.
 */
class VectorStore(private val chunkDao: ChunkDao) {

    suspend fun retrieve(
        queryEmbedding: FloatArray,
        relevantDocumentIds: List<Long>,
        topK: Int,
        query: String = ""
    ): List<ChunkEntity> {
        if (relevantDocumentIds.isEmpty()) return emptyList()

        val chunks = chunkDao.getAllChunksForDocumentIds(relevantDocumentIds)
        if (chunks.isEmpty()) return emptyList()

        var scoredChunks = chunks.mapNotNull { chunk ->
            val embeddingBytes = chunk.embedding ?: return@mapNotNull null
            val chunkEmbedding = FloatArrayConverter.toFloatArray(embeddingBytes) ?: return@mapNotNull null
            val score = cosineSimilarity(queryEmbedding, chunkEmbedding)
            Pair(chunk, score)
        }

        // Fallback to keyword matching if all scores are 0.0 (e.g. mock/empty embeddings)
        val allZeros = scoredChunks.all { it.second == 0f }
        if (allZeros && query.isNotEmpty()) {
            val keywords = query.lowercase().split(Regex("[^a-zA-Z0-9]+"))
                .filter { it.length > 2 && it !in listOf("what", "when", "where", "how", "this", "that", "there", "their", "them", "with", "from") }
            scoredChunks = chunks.map { chunk ->
                val score = keywords.count { chunk.text.lowercase().contains(it) }.toFloat()
                Pair(chunk, score)
            }.filter { it.second > 0f }
        }

        return scoredChunks
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
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
