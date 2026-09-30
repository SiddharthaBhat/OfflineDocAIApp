package com.siddu.offlinedocai.ai.embeddings

data class AvailableEmbeddingModel(
    val id: String,
    val name: String,
    val format: String,
    val downloadSize: String,
    val dimension: Int,
    val badge: String,
    val description: String,
    val fileName: String,
    val downloadUrl: String
)

object AvailableEmbeddingModels {
    val list: List<AvailableEmbeddingModel> = listOf(
        AvailableEmbeddingModel(
            id = "minilm-l6-v2-onnx",
            name = "all-MiniLM-L6-v2 (ONNX INT8)",
            format = "ONNX",
            downloadSize = "22 MB",
            dimension = 384,
            badge = "Recommended / 100% Parity",
            description = "High-precision 384-dim dense embedding model with full sentence-transformers parity. Runs natively via ONNX Runtime on all ARM devices.",
            fileName = "all-MiniLM-L6-v2-int8.onnx",
            downloadUrl = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
        ),
        AvailableEmbeddingModel(
            id = "bge-small-onnx",
            name = "BGE Small EN v1.5 (ONNX INT8)",
            format = "ONNX",
            downloadSize = "34 MB",
            dimension = 384,
            badge = "Top RAG Benchmark",
            description = "BAAI's state-of-the-art embedding model fine-tuned for semantic retrieval and question answering.",
            fileName = "bge-small-en-v1.5-int8.onnx",
            downloadUrl = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx"
        ),
        AvailableEmbeddingModel(
            id = "minilm-l6-v2-gguf",
            name = "all-MiniLM-L6-v2 (GGUF Q4_K_M)",
            format = "GGUF",
            downloadSize = "20 MB",
            dimension = 384,
            badge = "GGUF Format",
            description = "GGUF format embedding model for devices with full GGUF embedding backend support.",
            fileName = "all-MiniLM-L6-v2.Q4_K_M.gguf",
            downloadUrl = "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf"
        )
    )
}
