package com.siddu.offlinedocai.ai.queue

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener

/**
 * Parses user input containing multiple questions in various formats:
 * - JSON Array of strings: ["What is X?", "How does Y work?"]
 * - JSON Array of objects: [{"question": "What is X?"}, {"q": "How does Y work?"}]
 * - JSON Object with questions array: {"questions": ["What is X?", "How does Y work?"]}
 * - Plain newline-delimited text fallback
 */
object QuestionQueueParser {

    fun parseQuestions(rawInput: String): List<String> {
        val trimmed = rawInput.trim()
        if (trimmed.isEmpty()) return emptyList()

        // 1. Attempt JSON parsing
        val jsonQuestions = tryParseJson(trimmed)
        if (jsonQuestions.isNotEmpty()) {
            return jsonQuestions
        }

        // 2. Fallback to line-by-line parsing
        return parseLineByLine(trimmed)
    }

    private fun tryParseJson(input: String): List<String> {
        return try {
            val tokener = JSONTokener(input)
            val nextVal = tokener.nextValue()

            when (nextVal) {
                is JSONArray -> parseJsonArray(nextVal)
                is JSONObject -> {
                    // Check standard wrapper keys
                    val wrapperKeys = listOf("questions", "queue", "queries", "items", "prompts")
                    for (key in wrapperKeys) {
                        val arr = nextVal.optJSONArray(key)
                        if (arr != null) {
                            val parsed = parseJsonArray(arr)
                            if (parsed.isNotEmpty()) return parsed
                        }
                    }
                    // Check if the object itself is a single question
                    extractQuestionFromObject(nextVal)?.let { listOf(it) } ?: emptyList()
                }
                else -> emptyList()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun parseJsonArray(arr: JSONArray): List<String> {
        val results = mutableListOf<String>()
        for (i in 0 until arr.length()) {
            val item = arr.opt(i) ?: continue
            when (item) {
                is String -> {
                    val cleaned = cleanQuestion(item)
                    if (cleaned.isNotBlank()) results.add(cleaned)
                }
                is JSONObject -> {
                    extractQuestionFromObject(item)?.let {
                        if (it.isNotBlank()) results.add(it)
                    }
                }
                else -> {
                    val str = cleanQuestion(item.toString())
                    if (str.isNotBlank()) results.add(str)
                }
            }
        }
        return results
    }

    private fun extractQuestionFromObject(obj: JSONObject): String? {
        val candidateKeys = listOf("question", "q", "prompt", "query", "text", "content", "title")
        for (key in candidateKeys) {
            if (obj.has(key)) {
                val value = obj.optString(key, "")
                if (value.isNotBlank()) {
                    return cleanQuestion(value)
                }
            }
        }
        val keys = obj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val v = obj.optString(k, "")
            if (v.isNotBlank()) {
                return cleanQuestion(v)
            }
        }
        return null
    }

    private fun parseLineByLine(input: String): List<String> {
        return input.lines()
            .map { cleanQuestion(it) }
            .filter { it.isNotBlank() && !isJsonBracketLine(it) }
    }

    private fun isJsonBracketLine(line: String): Boolean {
        val t = line.trim()
        return t == "[" || t == "]" || t == "{" || t == "}" || t == "]," || t == "},"
    }

    fun cleanQuestion(raw: String): String {
        var str = raw.trim()
        // Strip markdown list bullets or numbers: "1. ", "1) ", "- ", "* "
        str = str.replace(Regex("""^(\d+[\.\)]|\-|\*)\s+"""), "")
        // Strip surrounding quotes
        if ((str.startsWith("\"") && str.endsWith("\"")) || (str.startsWith("'") && str.endsWith("'"))) {
            if (str.length >= 2) {
                str = str.substring(1, str.length - 1).trim()
            }
        }
        // Strip trailing comma
        if (str.endsWith(",")) {
            str = str.dropLast(1).trim()
        }
        // Strip quotes again if it was like: "Question",
        if ((str.startsWith("\"") && str.endsWith("\"")) || (str.startsWith("'") && str.endsWith("'"))) {
            if (str.length >= 2) {
                str = str.substring(1, str.length - 1).trim()
            }
        }
        return str.trim()
    }
}
