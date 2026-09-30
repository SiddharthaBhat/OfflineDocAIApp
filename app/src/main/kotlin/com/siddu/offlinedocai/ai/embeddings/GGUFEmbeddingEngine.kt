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
 * Universal On-Device Embedding Engine supporting both ONNX (via Microsoft ONNX Runtime)
 * and GGUF (via llama.cpp native bridge), with deterministic TF-IDF Edge Fallback.
 */
class GGUFEmbeddingEngine(private val context: Context) : EmbeddingModel {
    private val TAG = "GGUFEmbeddingEngine"

    private val fallbackEmbeddingDim = 768

    private var onnxEngine: ONNXEmbeddingEngine? = null
    private var ggufContext: LlamaContext? = null
    private var nativeEmbeddingDim: Int = 384

    init {
        loadActiveModel()
    }

    private fun loadActiveModel() {
        val modelDir = File(context.filesDir, "models/embeddings")
        if (!modelDir.exists()) modelDir.mkdirs()

        val prefs = context.getSharedPreferences("docai_models", Context.MODE_PRIVATE)
        val savedModelName = prefs.getString("selected_embedding_model", null)

        val allFiles = modelDir.listFiles()?.filter {
            it.extension.equals("onnx", ignoreCase = true) ||
            it.extension.equals("gguf", ignoreCase = true)
        } ?: emptyList()

        // Prioritize saved selection, otherwise ONNX models (most stable across chips), then GGUF
        val activeFile = allFiles.firstOrNull { it.name == savedModelName }
            ?: allFiles.firstOrNull { it.extension.equals("onnx", ignoreCase = true) }
            ?: allFiles.firstOrNull()

        if (activeFile == null) {
            Log.i(TAG, "No embedding model found on disk. Using edge fallback vectorizer (d=$fallbackEmbeddingDim).")
            return
        }

        Log.i(TAG, "Initializing active embedding model: ${activeFile.name}")
        if (activeFile.extension.equals("onnx", ignoreCase = true)) {
            try {
                onnxEngine = ONNXEmbeddingEngine(context, activeFile)
                nativeEmbeddingDim = 384
                Log.i(TAG, "ONNX embedding engine initialized successfully for ${activeFile.name}")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to initialize ONNX embedding engine: ${t.message}", t)
                onnxEngine = null
            }
        } else if (activeFile.extension.equals("gguf", ignoreCase = true)) {
            if (LlamaContext.isSupported()) {
                try {
                    ggufContext = LlamaContext(3, activeFile, 512, 512, 4, true)
                    Log.i(TAG, "GGUF embedding engine initialized successfully for ${activeFile.name}")
                } catch (t: Throwable) {
                    Log.e(TAG, "Failed to initialize GGUF embedding context: ${t.message}", t)
                    ggufContext = null
                }
            } else {
                Log.w(TAG, "LlamaContext not supported on this device.")
            }
        }
    }

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) {
        val clean = text.trim()
        if (clean.isEmpty()) {
            return@withContext FloatArray(nativeEmbeddingDim) { 0f }
        }

        // 1. Try ONNX engine (High Accuracy, zero JNI crash risk)
        onnxEngine?.let { engine ->
            try {
                val vec = engine.getEmbedding(clean)
                if (vec.isNotEmpty() && !vec.all { it == 0f }) {
                    nativeEmbeddingDim = vec.size
                    return@withContext vec
                }
            } catch (t: Throwable) {
                Log.w(TAG, "ONNX embedding error: ${t.message}", t)
            }
        }

        // 2. Try GGUF native engine
        ggufContext?.let { ctx ->
            try {
                val nativeVector = ctx.getEmbeddingVector(clean)
                if (nativeVector != null && nativeVector.isNotEmpty()) {
                    nativeEmbeddingDim = nativeVector.size
                    return@withContext l2Normalize(nativeVector)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "GGUF native embedding exception (${t.message}). Falling back.", t)
            }
        }

        // 3. Fallback: Sublinear TF-IDF Hash Vectorizer
        computeHashEmbedding(clean)
    }

    private fun computeHashEmbedding(text: String): FloatArray {
        val vector = FloatArray(fallbackEmbeddingDim) { 0f }
        val tokens = text.lowercase().split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 1 }

        if (tokens.isEmpty()) return vector

        val termFrequencies = mutableMapOf<String, Int>()
        for (token in tokens) {
            termFrequencies[token] = (termFrequencies[token] ?: 0) + 1
        }

        for ((term, count) in termFrequencies) {
            val weight = (1.0 + ln(count.toDouble())).toFloat()
            val h1 = abs(term.hashCode()) % fallbackEmbeddingDim
            val h2 = abs((term + "\u00a7").hashCode()) % fallbackEmbeddingDim
            val h3 = abs((term + term.length.toString()).hashCode()) % fallbackEmbeddingDim
            vector[h1] += weight
            vector[h2] += weight * 0.6f
            vector[h3] += weight * 0.3f
        }

        for (i in 0 until tokens.size - 1) {
            val bigram = "${tokens[i]}_${tokens[i + 1]}"
            val h = abs(bigram.hashCode()) % fallbackEmbeddingDim
            vector[h] += 1.5f
            val h2 = abs((bigram + "_b").hashCode()) % fallbackEmbeddingDim
            vector[h2] += 0.75f
        }

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
            onnxEngine?.close()
        } catch (ignored: Throwable) {}
        onnxEngine = null

        try {
            ggufContext?.release()
        } catch (ignored: Throwable) {}
        ggufContext = null
    }
}
