package com.siddu.offlinedocai.ai.llm

data class AvailableLlmModel(
    val id: String,
    val name: String,
    val parameterSize: String,
    val downloadSize: String,
    val badge: String,
    val description: String,
    val ramRequirement: String,
    val fileName: String,
    val downloadUrl: String
)

object AvailableLlmModels {
    val list: List<AvailableLlmModel> = listOf(
        AvailableLlmModel(
            id = "qwen-0.5b",
            name = "Qwen 2.5 0.5B Instruct",
            parameterSize = "0.5B",
            downloadSize = "390 MB",
            badge = "Fastest / Low RAM",
            description = "Ultra-fast response with minimal memory usage. Ideal for entry-level devices.",
            ramRequirement = "Min 2 GB RAM",
            fileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        AvailableLlmModel(
            id = "llama-1b",
            name = "Llama 3.2 1B Instruct",
            parameterSize = "1B",
            downloadSize = "700 MB",
            badge = "Standard Mobile",
            description = "Meta's lightweight model for general conversation and simple Q&A.",
            ramRequirement = "Min 2.5 GB RAM",
            fileName = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        AvailableLlmModel(
            id = "qwen-1.5b",
            name = "Qwen 2.5 1.5B Instruct",
            parameterSize = "1.5B",
            downloadSize = "1.0 GB",
            badge = "High Quality",
            description = "Top-tier sub-2B model. Exceptional document comprehension, table extraction & reasoning.",
            ramRequirement = "Min 3 GB RAM",
            fileName = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        AvailableLlmModel(
            id = "deepseek-r1-1.5b",
            name = "DeepSeek-R1 Distill 1.5B (Q6_K)",
            parameterSize = "1.5B",
            downloadSize = "1.5 GB",
            badge = "Advanced Reasoning",
            description = "DeepSeek-R1 reasoning distilled into 1.5B with high-precision Q6_K quantization.",
            ramRequirement = "Min 4 GB RAM",
            fileName = "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
            downloadUrl = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf"
        )
    )
}
