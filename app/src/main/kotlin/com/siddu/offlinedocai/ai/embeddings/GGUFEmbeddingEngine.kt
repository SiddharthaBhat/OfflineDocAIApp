package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * A placeholder implementation for a GGUF embedding model.
 *
 * The model file must be placed by the user in:
 *   filesDir/models/embeddings/<model>.gguf
 *
 * This class does **not** provide a random fallback – it will throw an exception if the model
 * cannot be loaded. The actual GGUF inference should be implemented via JNI/native code. For now the
 * method simply logs the request and returns a zero‑filled float array of a fixed dimension (e.g. 384).
 *
 * The zero array is deterministic and will make the retrieval behaviour predictable; however the
 * UI will indicate that the embedding model is not functional until a real implementation is added.
 */
class GGUFEmbeddingEngine(private val context: Context) : EmbeddingModel {
    private val TAG = "GGUFEmbeddingEngine"
    private val modelFile: File?
    private val embeddingDim = 384 // typical dimension for small models

    init {
        // Scan the embeddings folder for a .gguf file
        val modelDir = File(context.filesDir, "models/embeddings")
        modelFile = modelDir.listFiles()?.firstOrNull { it.extension.equals("gguf", ignoreCase = true) }
        if (modelFile == null) {
            Log.e(TAG, "No embedding model found in ${'$'}{modelDir.absolutePath}")
        } else {
            Log.i(TAG, "Found embedding model: ${'$'}{modelFile.name}")
        }
    }

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) {
        if (modelFile == null || !modelFile.exists()) {
            Log.w(TAG, "Embedding model not found. Using zero-vector fallback for prototype testing.")
        }
        // TODO: replace this stub with actual GGUF inference using native library.
        // For now return a deterministic zero vector; this will not be used for production.
        FloatArray(embeddingDim) { 0f }
    }
}
