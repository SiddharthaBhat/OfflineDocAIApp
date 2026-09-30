package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.nehuatl.llamacpp.LlamaContext
import java.io.File
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * High-Performance Dual-Mode On-Device Embedding Engine.
 *
 * 1. Native Real Mode: When a GGUF embedding model (e.g. all-MiniLM-L6-v2.Q4_K_M.gguf) is
 *    downloaded to `filesDir/models/embeddings/`, it loads and executes real dense neural
 *    embeddings directly on-device via librnllama.so JNI (LlamaContext with embedding=true).
 *
 * 2. Deterministic Edge Fallback: When no model is downloaded or native loading fails, it
 *    computes a deterministic L2-normalized sublinear 768-dimensional feature vector using
 *    an improved TF-IDF hash vectorizer with bigram projections (Pocket RAG / RAGdb mode).
 *    768-dim doubles the representation capacity vs the old 384-dim broken version.
 */
class GGUFEmbeddingEngine(private val context: Context) : EmbeddingModel {
    private val TAG = "GGUFEmbeddingEngine"

    // Raised to 768 for better representation fidelity in fallback mode
    private val fallbackEmbeddingDim = 768

    private val modelFile: File?
    private var embeddingContext: LlamaContext? = null
    private var nativeEmbeddingDim: Int = fallbackEmbeddingDim

    init {
        val modelDir = File(context.filesDir, "models/embeddings")
        if (!modelDir.exists()) modelDir.mkdirs()

        modelFile = modelDir.listFiles()?.firstOrNull {
            it.extension.equals("gguf", ignoreCase = true) ||
            it.extension.equals("onnx", ignoreCase = true)
        }

        if (modelFile == null) {
            Log.i(TAG, "No GGUF embedding model found. Using improved fallback vectorizer (d=$fallbackEmbeddingDim).")
        } else {
            Log.i(TAG, "Found local embedding model: ${modelFile.name}. Attempting native load.")
            tryLoadNativeEmbeddingModel(modelFile)
        }
    }

    /**
     * Attempt to initialize a native LlamaContext in embedding mode.
     * Uses nCtx=512, nBatch=512 (minimal — embedding models don't need large context).
     */
    private fun tryLoadNativeEmbeddingModel(model: File) {
        if (!LlamaContext.isSupported()) {
            Log.w(TAG, "Native llama library not supported on this device. Using fallback vectorizer.")
            return
        }
        try {
            // embedding=true is the critical flag — tells llama.cpp to enable pooled embeddings
            embeddingContext = LlamaContext(3, model, 512, 512, 4, true)
            Log.i(TAG, "Native GGUF embedding model loaded successfully: ${model.name}")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load native embedding model (${t.message}). Falling back to hash vectorizer.", t)
            embeddingContext = null
        }
    }

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) {
        val clean = text.trim()
        if (clean.isEmpty()) {
            return@withContext FloatArray(nativeEmbeddingDim) { 0f }
        }

        // 1. Try real native neural embedding
        val ctx = embeddingContext
        if (ctx != null) {
            try {
                val nativeVector = ctx.getEmbeddingVector(clean)
                if (nativeVector != null && nativeVector.isNotEmpty()) {
                    Log.d(TAG, "Native embedding produced ${nativeVector.size}-dim vector for: '${clean.take(50)}'")
                    nativeEmbeddingDim = nativeVector.size
                    return@withContext l2Normalize(nativeVector)
                } else {
                    Log.w(TAG, "Native embedding returned empty vector. Falling back to hash vectorizer.")
                }
            } catch (t: Throwable) {
                Log.w(TAG, "Native embedding threw exception (${t.message}). Falling back to hash vectorizer.", t)
            }
        }

        // 2. Improved fallback: Sublinear TF-IDF Hash Vectorizer (d=768, unigrams + bigrams + trigrams)
        computeHashEmbedding(clean)
    }

    /**
     * Improved deterministic fallback: 768-dim feature-hashed TF-IDF vector with:
     * - Unigram sublinear TF weighting: w = 1 + ln(count)
     * - Dual-hash unigram projection (reduces hash collision impact)
     * - Bigram phrase sensitivity features
     * - Trigram local context windows for short queries
     * - L2 normalization so cosine similarity is well-defined
     */
    private fun computeHashEmbedding(text: String): FloatArray {
        val vector = FloatArray(fallbackEmbeddingDim) { 0f }
        val tokens = text.lowercase().split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 1 }

        if (tokens.isEmpty()) return vector

        // Unigram TF with sublinear weighting
        val termFrequencies = mutableMapOf<String, Int>()
        for (token in tokens) {
            termFrequencies[token] = (termFrequencies[token] ?: 0) + 1
        }

        for ((term, count) in termFrequencies) {
            val weight = (1.0 + ln(count.toDouble())).toFloat()
            // Two hash projections per term to reduce collision impact
            val h1 = abs(term.hashCode()) % fallbackEmbeddingDim
            val h2 = abs((term + "\u00a7").hashCode()) % fallbackEmbeddingDim
            val h3 = abs((term + term.length.toString()).hashCode()) % fallbackEmbeddingDim
            vector[h1] += weight
            vector[h2] += weight * 0.6f
            vector[h3] += weight * 0.3f
        }

        // Bigram features — capture phrase-level semantics
        for (i in 0 until tokens.size - 1) {
            val bigram = "${tokens[i]}_${tokens[i + 1]}"
            val h = abs(bigram.hashCode()) % fallbackEmbeddingDim
            vector[h] += 1.5f
            // Second hash for bigram
            val h2 = abs((bigram + "_b").hashCode()) % fallbackEmbeddingDim
            vector[h2] += 0.75f
        }

        // Trigram features — useful for distinguishing e.g. "select plant for grafting"
        for (i in 0 until tokens.size - 2) {
            val trigram = "${tokens[i]}_${tokens[i + 1]}_${tokens[i + 2]}"
            val h = abs(trigram.hashCode()) % fallbackEmbeddingDim
            vector[h] += 0.8f
        }

        return l2Normalize(vector)
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSquares = 0.0
        for (v in vector) sumSquares += (v * v).toDouble()
        val norm = sqrt(sumSquares)
        if (norm > 0.0) {
            for (i in vector.indices) vector[i] = (vector[i] / norm).toFloat()
        }
        return vector
    }

    fun release() {
        try {
            embeddingContext?.release()
        } catch (t: Throwable) {
            Log.w(TAG, "Error releasing embedding context: ${t.message}")
        }
        embeddingContext = null
    }
}
