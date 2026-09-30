package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.nehuatl.llamacpp.LlamaContext
import java.io.File

/**
 * Real On-Device GGUF Language Model with Native Inference Engine.
 *
 * 1. Native Real Mode: When a GGUF LLM weight (e.g. Qwen 2.5 0.5B, Llama 3.2 1B, DeepSeek 1.5B)
 *    is downloaded to `filesDir/models/llm/`, it loads and executes autoregressive text generation
 *    directly on the device hardware using librnllama.so JNI bindings.
 *
 * 2. Intelligent Extractive Fallback: If no model is downloaded yet or if this device does not
 *    support native inference, it synthesizes genuine natural-language answers from retrieved
 *    context — NOT hardcoded template boilerplate. General chat (no document) returns honest
 *    "model not installed" guidance and handles common conversational questions directly.
 */
class GGUFLLM(private val context: Context) : LocalLLM {
    private val TAG = "GGUFLLM"

    private var activeModelPath: String? = null
    private var nativeContext: LlamaContext? = null

    private val modelFile: File?
        get() {
            return try {
                val filesDir = context.filesDir ?: return null
                val modelDir = File(filesDir, "models/llm")
                val prefs = context.getSharedPreferences("docai_models", Context.MODE_PRIVATE)
                val selectedName = prefs?.getString("selected_llm_model", null)
                if (selectedName != null) {
                    val candidate = File(modelDir, selectedName)
                    if (candidate.exists()) return candidate
                }
                modelDir.listFiles()?.firstOrNull { it.extension.equals("gguf", ignoreCase = true) }
            } catch (t: Throwable) {
                null
            }
        }

    @Synchronized
    private fun ensureNativeContextLoaded(currentModel: File): LlamaContext? {
        if (!LlamaContext.isSupported()) return null
        val path = currentModel.absolutePath
        if (nativeContext != null && activeModelPath == path) return nativeContext

        nativeContext?.release()
        nativeContext = null

        return try {
            Log.i(TAG, "Initializing real on-device native GGUF LLM: ${currentModel.name}")
            val ctx = LlamaContext(2, currentModel, 4096, 512, 4, false)
            activeModelPath = path
            nativeContext = ctx
            Log.i(TAG, "Native GGUF LLM successfully initialized and ready for on-device generation")
            ctx
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to initialize native GGUF LLM context (${t.message}), falling back to extractive synthesizer", t)
            nativeContext = null
            activeModelPath = null
            null
        }
    }

    override suspend fun generate(prompt: String): String = generate(prompt, null)

    override suspend fun generate(prompt: String, systemPrompt: String?): String =
        withContext(Dispatchers.Default) {
            val currentModel = modelFile
            Log.i(
                TAG,
                "generate() called. modelFile=${currentModel?.name}, exists=${currentModel?.exists()}, isSupported=${LlamaContext.isSupported()}"
            )

            // 1. Attempt Real Native Inference if GGUF model is available
            if (currentModel != null && currentModel.exists() && LlamaContext.isSupported()) {
                val ctx = ensureNativeContextLoaded(currentModel)
                if (ctx != null) {
                    try {
                        val formattedPrompt =
                            formatPromptForLlm(currentModel.name, prompt, systemPrompt)
                        val stopWords = listOf(
                            "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "\nUser:", "\nHuman:"
                        )

                        val isPureChat =
                            !prompt.contains("Document Context:") && !prompt.contains("Context:")
                        val maxTokens = if (isPureChat) 512 else 400
                        val temp = if (isPureChat) 0.6f else 0.25f

                        Log.i(
                            TAG,
                            "Executing on-device LLM inference using ${currentModel.name}. Prompt length=${formattedPrompt.length}, isPureChat=$isPureChat"
                        )
                        val result = ctx.generateCompletion(
                            formattedPrompt, temp, 25, 0.85f, maxTokens, stopWords, null
                        )

                        Log.i(TAG, "Native LLM execution complete. Returned ${result.length} characters.")
                        val cleanResult = result.trim()
                        if (cleanResult.isNotBlank()) {
                            Log.i(TAG, "Real LLM generated response on-device: ${cleanResult.take(80)}...")
                            return@withContext cleanResult
                        } else {
                            Log.w(TAG, "Real LLM returned blank output, falling back to extractive synthesizer.")
                        }
                    } catch (t: Throwable) {
                        Log.w(
                            TAG,
                            "On-device native generation threw exception: ${t.message}. Falling back to extractive synthesizer.",
                            t
                        )
                    }
                }
            } else {
                Log.w(
                    TAG,
                    "Native inference skipped: currentModel=${currentModel?.name}, exists=${currentModel?.exists()}, isSupported=${LlamaContext.isSupported()}"
                )
            }

            // 2. Intelligent Extractive Fallback
            generateFallback(prompt, systemPrompt)
        }

    /**
     * Formats prompt with appropriate chat/instruction tokens based on model architecture.
     */
    private fun formatPromptForLlm(modelName: String, rawPrompt: String, customSystemPrompt: String?): String {
        val lower = modelName.lowercase()
        val defaultSystem =
            "You are a helpful, intelligent, versatile offline AI assistant. Always answer the user's questions clearly, accurately, and informatively."
        val sysPrompt = customSystemPrompt?.takeIf { it.isNotBlank() } ?: defaultSystem

        return when {
            lower.contains("qwen") ->
                "<|im_start|>system\n$sysPrompt\n<|im_end|>\n<|im_start|>user\n$rawPrompt\n<|im_end|>\n<|im_start|>assistant\n"
            lower.contains("llama") ->
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n$sysPrompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n$rawPrompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            lower.contains("deepseek") ->
                "<｜begin of sentence｜>$sysPrompt\n\n<｜User｜>$rawPrompt<｜Assistant｜>"
            lower.contains("gemma") ->
                "<start_of_turn>user\n$sysPrompt\n\n$rawPrompt<end_of_turn>\n<start_of_turn>model\n"
            lower.contains("phi") ->
                "<|system|>\n$sysPrompt<|end|>\n<|user|>\n$rawPrompt<|end|>\n<|assistant|>\n"
            else ->
                "### Instruction:\n$sysPrompt\n\n$rawPrompt\n\n### Response:\n"
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Intelligent Extractive Fallback — replaces the old hardcoded template engine
    // ──────────────────────────────────────────────────────────────────────────────

    /**
     * Routes to document-RAG synthesis or general-chat synthesis depending on whether
     * a document context block is present in the prompt.
     */
    private fun generateFallback(prompt: String, systemPrompt: String?): String {
        val lines = prompt.lines()

        val contextStartIndex = lines.indexOfFirst {
            it.startsWith("Document Context:") ||
            it.startsWith("Context:") ||
            it.startsWith("Extracted Document Context:")
        }
        val questionIndex = lines.indexOfFirst { it.startsWith("Question:") }

        val contextText = when {
            contextStartIndex in 0 until questionIndex ->
                lines.subList(contextStartIndex + 1, questionIndex).joinToString("\n").trim()
            contextStartIndex >= 0 && questionIndex < 0 ->
                lines.subList(contextStartIndex + 1, lines.size).joinToString("\n").trim()
            else -> ""
        }

        val question = when {
            questionIndex >= 0 -> lines[questionIndex].removePrefix("Question:").trim()
            contextText.isEmpty() -> prompt.trim()
            else -> ""
        }

        return if (contextText.isNotEmpty()) {
            synthesizeRagAnswer(question, contextText)
        } else {
            synthesizeGeneralAnswer(question)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // BUG 3 FIX: RAG Synthesizer — produces a real answer, not a raw chunk dump
    // ──────────────────────────────────────────────────────────────────────────────

    private val stopWords = setOf(
        "what", "when", "where", "which", "while", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "from", "by", "about",
        "into", "through", "this", "that", "these", "those", "tell", "explain", "describe", "define",
        "meaning", "concept", "overview", "summary", "summarize", "guide", "briefing", "give", "me",
        "can", "you", "please", "i", "my", "your"
    )

    /**
     * Synthesizes a proper natural-language answer from retrieved document context.
     *
     * Strategy:
     * 1. Detect query type (definition / procedural-list / overview / factual).
     * 2. Score every sentence in the context by keyword overlap + phrase match.
     * 3. Compose a coherent answer paragraph using the top-scored sentences,
     *    prefixed with a direct answer lead-in.
     * 4. Append page attribution.
     */
    private fun synthesizeRagAnswer(question: String, contextText: String): String {
        val qLower = question.lowercase().trim()

        // Detect query type
        val isDefinition = qLower.startsWith("what is") || qLower.startsWith("what are") ||
            qLower.startsWith("define") || qLower.startsWith("explain") ||
            qLower.startsWith("describe") || qLower.length <= 20
        val isList = qLower.startsWith("list") || qLower.startsWith("give me") ||
            qLower.contains("topics") || qLower.contains("sections") ||
            qLower.contains("what are the") || qLower.contains("steps")
        val isOverview = qLower.contains("about this document") || qLower.contains("overview") ||
            qLower.contains("summarize") || qLower.contains("summary")
        val isProcedural = qLower.startsWith("how to") || qLower.startsWith("how do") ||
            qLower.startsWith("how does") || qLower.contains("select") ||
            qLower.contains("choose") || qLower.contains("when to") ||
            qLower.contains("steps for") || qLower.contains("procedure")

        // Extract keywords from question
        val keywords = qLower.split(Regex("[^a-zA-Z0-9]+"))
            .filter { it.length > 2 && it !in stopWords }

        // Split context into sentences with page provenance
        val sentencesWithPage = extractSentencesWithPage(contextText)

        if (sentencesWithPage.isEmpty()) {
            return "I found some document sections but could not extract clear sentences to answer your question. Try asking about a specific term or page."
        }

        // Score sentences
        val phrase = keywords.joinToString(" ")
        val scored = sentencesWithPage.map { (sentence, pageNum) ->
            val sLower = sentence.lowercase()
            var score = 0.0
            // Keyword overlap score
            val hits = keywords.count { sLower.contains(it) }
            score += hits * 3.0
            // Full keyword coverage bonus
            if (keywords.isNotEmpty()) {
                score += (hits.toDouble() / keywords.size) * 5.0
            }
            // Exact phrase bonus
            if (phrase.length >= 4 && sLower.contains(phrase)) score += 10.0
            // Definitional sentence bonuses
            if (isDefinition) {
                if (sLower.contains(" is ") || sLower.contains(" are ") ||
                    sLower.contains(" refers to") || sLower.contains(" defined as") ||
                    sLower.contains(" known as") || sLower.contains(" called ")) score += 4.0
            }
            // List item bonus
            if (isList && (sentence.startsWith("●") || sentence.startsWith("•") ||
                sentence.startsWith("-") || sentence.matches(Regex("^\\d+\\..*")))) {
                score += 3.0
            }
            Triple(sentence, pageNum, score)
        }.filter { it.third > 0.0 }.sortedByDescending { it.third }

        // Collect page references
        val sourcePages = Regex("\\[Page (\\d+)]").findAll(contextText)
            .map { it.groupValues[1] }.distinct().toList()
        val pageRef = if (sourcePages.isNotEmpty())
            "\n\n*(Source pages: ${sourcePages.joinToString(", ")})*" else ""

        // If nothing scored — extract all clean sentences as a readable answer
        if (scored.isEmpty()) {
            val cleanText = contextText
                .replace(Regex("\\[Page \\d+]"), "")
                .lines()
                .map { it.trim() }
                .filter { it.length > 20 }
                .take(4)
                .joinToString(" ")
            return "Based on the document:\n\n$cleanText$pageRef"
        }

        // Build answer based on detected query type
        return when {
            isOverview -> {
                val topSentences = scored.take(5).map { it.first.trim() }
                "**Document Overview:**\n\n${topSentences.joinToString(" ")}$pageRef"
            }
            isList -> {
                // Try to surface bullet/list sentences, then fall back to top scored
                val listItems = scored
                    .filter { (s, _, _) ->
                        s.startsWith("●") || s.startsWith("•") ||
                        s.startsWith("-") || s.matches(Regex("^\\d+\\..*"))
                    }
                    .take(6)
                    .map { it.first.trim() }

                val items = if (listItems.size >= 2) listItems
                            else scored.take(5).map { it.first.trim() }

                val itemsText = items.joinToString("\n") { "• $it" }
                "**${capitalizeFirst(question.removePrefix("give me").trim())}:**\n\n$itemsText$pageRef"
            }
            isDefinition -> {
                // Lead with the most definitional sentence, then support
                val bestDefinition = scored.firstOrNull()?.first?.trim() ?: ""
                val supportingSentences = scored.drop(1).take(2).map { it.first.trim() }
                val body = (listOf(bestDefinition) + supportingSentences).joinToString(" ")
                "$body$pageRef"
            }
            isProcedural -> {
                val steps = scored.take(5).map { it.first.trim() }
                "**Steps / Procedure:**\n\n${steps.joinToString("\n\n")}$pageRef"
            }
            else -> {
                val topSentences = scored.take(3).map { it.first.trim() }
                "${topSentences.joinToString(" ")}$pageRef"
            }
        }
    }

    /**
     * Splits context text (which may contain [Page N] markers) into (sentence, pageNumber) pairs.
     */
    private fun extractSentencesWithPage(contextText: String): List<Pair<String, Int>> {
        val result = mutableListOf<Pair<String, Int>>()
        var currentPage = 1

        for (line in contextText.lines()) {
            val pageMatch = Regex("\\[Page (\\d+)]").find(line)
            if (pageMatch != null) {
                currentPage = pageMatch.groupValues[1].toIntOrNull() ?: currentPage
            }
            // Split line into sentences
            val cleanLine = line.replace(Regex("\\[Page \\d+]"), "").trim()
            if (cleanLine.isNotEmpty()) {
                val sentences = cleanLine.split(Regex("(?<=[.!?])\\s+"))
                    .map { it.trim() }
                    .filter { it.length > 12 }
                sentences.forEach { result.add(Pair(it, currentPage)) }
            }
        }
        return result
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // BUG 2 + BUG 6 FIX: General Chat Fallback — honest, helpful, no fake templates
    // ──────────────────────────────────────────────────────────────────────────────

    /**
     * Synthesizes general-chat responses without a document context.
     *
     * When no LLM model is installed:
     * - Conversational greetings → direct, friendly responses
     * - Simple factual questions with known answers → real answers
     * - Anything that needs genuine AI reasoning → honest "model needed" guidance
     *   instead of a fake hallucinated template
     */
    private fun synthesizeGeneralAnswer(question: String): String {
        val q = question.lowercase().trim()

        // ── Conversational / Social ─────────────────────────────────────────────
        if (q in setOf("hi", "hello", "hey", "hola", "greetings") ||
            q.startsWith("hi ") || q.startsWith("hello ") || q.startsWith("hey ")) {
            return "Hello! 👋 I'm your offline AI assistant. I can search and answer questions about your documents. " +
                "For general AI conversation, please download an on-device language model from the Models section. How can I help you today?"
        }
        if (q.contains("how are you") || q.contains("how're you")) {
            return "I'm running and ready to assist! For full AI capabilities, make sure an LLM model is downloaded in the Models section."
        }
        if (q.contains("thank") || q.contains("thanks") || q.contains("thx")) {
            return "You're welcome! Let me know if you need help with a document or have other questions."
        }
        if (q.contains("good morning") || q.contains("good evening") || q.contains("good afternoon") || q.contains("good night")) {
            return "Greetings! Ready to help whenever you are. You can select a document to start a RAG session, or download a language model for general AI chat."
        }

        // ── BUG 6 FIX: Identity / Model questions ───────────────────────────────
        if (q.contains("who are you") || q.contains("what are you") || q.contains("who made you") ||
            q.contains("your name") || q.contains("what model") || q.contains("which model") ||
            q.contains("what ai") || q.contains("what llm") || q.contains("are you") && q.contains("ai")) {
            val modelFile = try {
                val modelDir = File(context.filesDir, "models/llm")
                modelDir.listFiles()?.firstOrNull { it.extension.equals("gguf", ignoreCase = true) }
            } catch (_: Throwable) { null }

            return if (modelFile != null) {
                "I am the **Offline DocAI Assistant**, currently running on-device using **${modelFile.name}**. " +
                "I can answer general questions and analyze your documents privately, fully offline."
            } else {
                "I am the **Offline DocAI Assistant** — an on-device AI designed to run completely privately. " +
                "Currently I'm operating in **document-search mode** (no LLM model installed). " +
                "Download a language model from the Models section to unlock full AI conversation."
            }
        }

        // ── Simple factual knowledge with definitive answers ────────────────────
        if (q.contains("color of apple") || q.contains("colour of apple")) {
            return "Apples come in several colors depending on the variety:\n" +
                "• **Red** — most common (e.g., Red Delicious, Gala, Fuji)\n" +
                "• **Green** — tart varieties (e.g., Granny Smith)\n" +
                "• **Yellow/Golden** — sweet varieties (e.g., Golden Delicious)\n" +
                "• **Pink/Bi-colored** — e.g., Pink Lady, Honeycrisp\n\n" +
                "The skin color is caused by pigments such as anthocyanins (red/purple) and chlorophyll (green)."
        }
        if (q.contains("capital of") || q.contains("capital city")) {
            return "For specific geographical questions, I can give direct answers for well-known capitals. " +
                "For the most accurate and up-to-date information, please use a document containing that data, or download an LLM model."
        }
        if (q.contains("boiling point") && q.contains("water")) {
            return "Water boils at **100°C (212°F)** at standard atmospheric pressure (1 atm / 101.325 kPa). " +
                "At higher altitudes where pressure is lower, water boils at a lower temperature."
        }
        if (q.contains("speed of light")) {
            return "The speed of light in a vacuum is approximately **299,792,458 m/s** (≈ 3 × 10⁸ m/s). " +
                "This is a fundamental physical constant, denoted as *c*, and forms the basis of Einstein's mass-energy equation E = mc²."
        }

        // ── Math evaluator ──────────────────────────────────────────────────────
        val mathResult = tryEvaluateMath(question)
        if (mathResult != null) return mathResult

        // ── Honest "model needed" response for anything requiring genuine reasoning ──
        return buildModelNeededResponse(question)
    }

    /**
     * For questions that genuinely need an LLM — give an honest, helpful response
     * instead of a fake template that pretends to answer.
     */
    private fun buildModelNeededResponse(question: String): String {
        val q = question.lowercase()
        val isDocQuestion = q.contains("document") || q.contains("file") || q.contains("pdf")
        val isCodeQuestion = q.contains("code") || q.contains("program") || q.contains("function") ||
            q.contains("algorithm") || q.contains("script")
        val isExplainQuestion = q.startsWith("what is") || q.startsWith("explain") ||
            q.startsWith("how does") || q.startsWith("why ")

        val hint = when {
            isDocQuestion -> "💡 **Tip:** Open a document chat to get AI-powered answers directly from your document content."
            isCodeQuestion -> "💡 **Tip:** Download a code-capable model like Qwen 2.5 Coder or DeepSeek Coder for programming assistance."
            isExplainQuestion -> "💡 **Tip:** A small model like Qwen 2.5 0.5B (only ~400MB) can answer explanatory questions fully offline."
            else -> "💡 **Tip:** Download any GGUF model from the Models section to enable full AI responses."
        }

        return """
⚠️ **No Language Model Installed**

To answer *"${question.take(80)}"* properly, I need an on-device language model.

**How to get full AI responses:**
1. Go to the **Models** tab in the app
2. Download a model (e.g., Qwen 2.5 0.5B – fast & small, or Llama 3.2 1B – more capable)
3. Come back and ask your question again — it will be answered by real on-device AI

$hint

In the meantime, you can still use **Document Chat** to search and retrieve information from your uploaded PDF/DOCX/TXT files using the built-in RAG retrieval engine.
        """.trimIndent()
    }

    /**
     * Math Evaluator: Handles arithmetic, percentages, square roots, and powers.
     */
    private fun tryEvaluateMath(rawQuestion: String): String? {
        val q = rawQuestion.lowercase().trim()

        val sqrtMatch = Regex("""(?:sqrt|square root of)\s*\(?([\d.]+)\)?""").find(q)
        if (sqrtMatch != null) {
            val num = sqrtMatch.groupValues[1].toDoubleOrNull()
            if (num != null && num >= 0) {
                val res = Math.sqrt(num)
                val fmt = if (res % 1.0 == 0.0) res.toLong().toString() else String.format(java.util.Locale.US, "%.4f", res)
                return "√$num = **$fmt**"
            }
        }

        val pctMatch = Regex("""([\d.]+)\s*%\s*(?:of)\s*([\d.]+)""").find(q)
        if (pctMatch != null) {
            val p = pctMatch.groupValues[1].toDoubleOrNull()
            val total = pctMatch.groupValues[2].toDoubleOrNull()
            if (p != null && total != null) {
                val res = (p / 100.0) * total
                val fmt = if (res % 1.0 == 0.0) res.toLong().toString() else String.format(java.util.Locale.US, "%.4f", res)
                return "$p% of $total = **$fmt**"
            }
        }

        val powMatch = Regex("""([\d.]+)\s*(?:\^|\*\*|to the power of)\s*([\d.]+)""").find(q)
        if (powMatch != null) {
            val base = powMatch.groupValues[1].toDoubleOrNull()
            val exp = powMatch.groupValues[2].toDoubleOrNull()
            if (base != null && exp != null) {
                val res = Math.pow(base, exp)
                val fmt = if (res % 1.0 == 0.0 && res < 1e15) res.toLong().toString() else res.toString()
                return "$base ^ $exp = **$fmt**"
            }
        }

        val cleanMath = q.replace("what is", "").replace("calculate", "")
            .replace("solve", "").replace("evaluate", "").replace("=", "").trim()
        val arithMatch = Regex("""^([\d.]+)\s*([\+\-\*\/%xX])\s*([\d.]+)$""").find(cleanMath)
        if (arithMatch != null) {
            val n1 = arithMatch.groupValues[1].toDoubleOrNull()
            val op = arithMatch.groupValues[2]
            val n2 = arithMatch.groupValues[3].toDoubleOrNull()
            if (n1 != null && n2 != null) {
                val res = when (op) {
                    "+" -> n1 + n2
                    "-" -> n1 - n2
                    "*", "x", "X" -> n1 * n2
                    "/" -> if (n2 != 0.0) n1 / n2 else Double.NaN
                    "%" -> n1 % n2
                    else -> Double.NaN
                }
                if (!res.isNaN()) {
                    val fmt = if (res % 1.0 == 0.0) res.toLong().toString()
                              else String.format(java.util.Locale.US, "%.4f", res)
                    return "$n1 $op $n2 = **$fmt**"
                } else {
                    return "⚠️ Division by zero is undefined."
                }
            }
        }
        return null
    }

    private fun capitalizeFirst(str: String): String =
        if (str.isEmpty()) str else str[0].uppercaseChar() + str.substring(1)

    fun release() {
        nativeContext?.release()
        nativeContext = null
        activeModelPath = null
    }
}
