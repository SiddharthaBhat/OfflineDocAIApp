package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import org.junit.Assert.*
import org.junit.Test

class AvailableLlmModelsTest {

    @Test
    fun testCatalogHasRequiredModels() {
        val models = AvailableLlmModels.list
        assertTrue("Catalog should have at least 4 models", models.size >= 4)

        val ids = models.map { it.id }
        assertTrue("Contains 0.5B model", ids.contains("qwen-0.5b"))
        assertTrue("Contains 1B model", ids.contains("llama-1b"))
        assertTrue("Contains 1.5B (1GB) model", ids.contains("qwen-1.5b"))
        assertTrue("Contains DeepSeek R1 1.5B (1.5GB) model", ids.contains("deepseek-r1-1.5b"))

        // Check uniqueness of IDs and file names
        assertEquals("All model IDs must be unique", models.map { it.id }.distinct().size, models.size)
        assertEquals("All model fileNames must be unique", models.map { it.fileName }.distinct().size, models.size)
    }

    @Test
    fun testModelDownloadUrlsAndFiles() {
        for (model in AvailableLlmModels.list) {
            assertTrue("URL must start with https://huggingface.co", model.downloadUrl.startsWith("https://huggingface.co/"))
            assertTrue("File name must end with .gguf", model.fileName.endsWith(".gguf", ignoreCase = true))
            assertTrue("Parameter size must not be empty", model.parameterSize.isNotBlank())
            assertTrue("RAM requirement must not be empty", model.ramRequirement.isNotBlank())
            assertTrue("Badge must not be empty", model.badge.isNotBlank())
            assertTrue("Description must not be empty", model.description.isNotBlank())
        }
    }

    @Test
    fun testSpecificTargetModelSpecs() {
        val qwen15 = AvailableLlmModels.list.find { it.id == "qwen-1.5b" }
        assertNotNull("Qwen 1.5B model should exist", qwen15)
        assertEquals("qwen2.5-1.5b-instruct-q4_k_m.gguf", qwen15!!.fileName)
        assertTrue(qwen15.downloadSize.contains("1.0 GB") || qwen15.downloadSize.contains("990 MB"))

        val deepseekR1 = AvailableLlmModels.list.find { it.id == "deepseek-r1-1.5b" }
        assertNotNull("DeepSeek R1 model should exist", deepseekR1)
        assertEquals("DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf", deepseekR1!!.fileName)
        assertTrue(deepseekR1.downloadSize.contains("1.5 GB"))
    }
}
