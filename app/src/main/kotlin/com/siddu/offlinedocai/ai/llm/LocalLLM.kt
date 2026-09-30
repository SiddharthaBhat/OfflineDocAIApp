package com.siddu.offlinedocai.ai.llm

/**
 * Simple interface representing a language model capable of generating a response for a given prompt.
 * Implementations run on a background dispatcher and either use real on-device GGUF inference
 * or an intelligent extractive fallback when no model is installed.
 */
interface LocalLLM {
    /**
     * Generate a response for [prompt]. The function runs on a background dispatcher and returns the
     * generated text, or throws an exception if generation fails.
     */
    suspend fun generate(prompt: String): String

    /**
     * Generate a response with an explicit [systemPrompt] behavioral instruction.
     *
     * Implementations MUST override this to properly incorporate the system prompt.
     * The default here simply delegates to generate(prompt) — concrete classes should
     * override to correctly inject the system instruction into the formatted prompt.
     */
    suspend fun generate(prompt: String, systemPrompt: String?): String = generate(prompt)
}
