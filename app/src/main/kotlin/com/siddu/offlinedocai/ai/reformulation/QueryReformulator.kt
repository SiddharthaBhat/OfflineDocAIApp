package com.siddu.offlinedocai.ai.reformulation

import android.util.Log
import com.siddu.offlinedocai.ai.profiler.DocumentProfile
import com.siddu.offlinedocai.ai.profiler.DocumentSection

/**
 * High-level intent classification for RAG query processing.
 */
sealed class QueryIntent {
    data class PageNavigation(val targetPage: Int) : QueryIntent()
    object GlobalSummary : QueryIntent()
    data class Definitional(val coreTerm: String, val rewrittenQuery: String) : QueryIntent()
    data class Procedural(val coreAction: String, val rewrittenQuery: String) : QueryIntent()
    data class DomainSpecific(val originalQuery: String) : QueryIntent()
}

/**
 * Result of query analysis and intent reformulation.
 */
data class ReformulationResult(
    val intent: QueryIntent,
    val searchString: String,
    val isRewritten: Boolean
)

/**
 * Context-Aware Query Reformulator.
 *
 * Implements deterministic intent classification and domain expansion grounded in the Document Knowledge Profile.
 */
object QueryReformulator {

    private const val TAG = "DocAI_QueryReformulator"

    private fun logInfo(msg: String) {
        try {
            Log.i(TAG, msg)
        } catch (_: Throwable) {
            // Safe fallback for local JVM unit testing
        }
    }

    /**
     * Classifies user query and produces an optimal search string.
     */
    fun classifyAndRewrite(
        rawQuery: String,
        profile: DocumentProfile? = null
    ): ReformulationResult {
        val clean = rawQuery.trim().lowercase().trimEnd('?', '!', '.', ' ')

        // 1. Layer 1: Direct Page Navigation Intent
        val pageMatch = Regex("^(?:go to|show me|show|open|view|take me to)?\\s*(?:page|p\\.?)\\s*(\\d+)$").find(clean)
        if (pageMatch != null) {
            val pageNum = pageMatch.groupValues[1].toIntOrNull() ?: 1
            logInfo("Classified query as PageNavigation: targetPage=$pageNum")
            return ReformulationResult(
                intent = QueryIntent.PageNavigation(pageNum),
                searchString = clean,
                isRewritten = false
            )
        }

        // 2. Layer 1: Global Summary Intent (Strict Dominance Check)
        if (isGlobalSummaryQuery(clean)) {
            logInfo("Classified query as GlobalSummary: bypass vector search")
            return ReformulationResult(
                intent = QueryIntent.GlobalSummary,
                searchString = clean,
                isRewritten = false
            )
        }

        // 3. Layer 2: Definitional Intent
        val defPrefixes = listOf(
            "what is", "what are", "define", "definition of", "explain",
            "describe", "tell me about", "meaning of", "concept of", "overview of"
        )
        for (prefix in defPrefixes) {
            if (clean.startsWith(prefix)) {
                val term = clean.removePrefix(prefix).trim().removePrefix("the").trim()
                if (term.isNotBlank()) {
                    logInfo("Classified Definitional: term='$term'")
                    val rewritten = "$term definition"
                    return ReformulationResult(
                        intent = QueryIntent.Definitional(term, rewritten),
                        searchString = rewritten,
                        isRewritten = true
                    )
                }
            }
        }

        if (clean.endsWith(" meaning") || clean.endsWith(" definition")) {
            val term = clean.substringBeforeLast(" ").trim()
            if (term.isNotBlank()) {
                logInfo("Classified Definitional (suffix): term='$term'")
                val rewritten = "$term definition"
                return ReformulationResult(
                    intent = QueryIntent.Definitional(term, rewritten),
                    searchString = rewritten,
                    isRewritten = true
                )
            }
        }

        // Single isolated technical noun query (e.g. "grafting", "rootstock")
        val words = clean.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size in 1..2 && !listOf("how", "why", "when", "where", "can", "is", "do", "what").any { clean.startsWith(it) }) {
            val term = clean
            logInfo("Classified Definitional (single noun): term='$term'")
            val rewritten = "$term definition"
            return ReformulationResult(
                intent = QueryIntent.Definitional(term, rewritten),
                searchString = rewritten,
                isRewritten = true
            )
        }

        // 4. Layer 2: Procedural Intent
        val procPrefixes = listOf(
            "how to", "how do", "how does", "steps to", "steps for", "procedure for",
            "why do", "why is", "when to", "when should", "when can", "where to",
            "troubleshooting", "tools needed", "tools for"
        )
        for (prefix in procPrefixes) {
            if (clean.startsWith(prefix)) {
                val action = clean.removePrefix(prefix).trim()
                logInfo("Classified Procedural: action='$action'")
                val rewritten = "$action procedure guide"
                return ReformulationResult(
                    intent = QueryIntent.Procedural(action, rewritten),
                    searchString = rewritten,
                    isRewritten = true
                )
            }
        }

        // 5. Layer 3: Safe Pass-Through Fallback (Zero Mutation)
        logInfo("No specific intent matched. Emitting raw query pass-through: '$rawQuery'")
        return ReformulationResult(
            intent = QueryIntent.DomainSpecific(rawQuery),
            searchString = rawQuery,
            isRewritten = false
        )
    }

    /**
     * Strict Dominance Check for Global Summary Queries.
     * Prevents queries like "overview of cambium" or "summary of techniques" from falsely matching.
     */
    fun isGlobalSummaryQuery(clean: String): Boolean {
        val exactGlobalPhrases = setOf(
            "main points", "key takeaways", "explain document",
            "what is this document about", "what is this file about", "what is this doc about", "what's this doc about",
            "what is this about", "what's this about", "what does this document say", "what does this file say",
            "summarize document", "summary of document", "document summary",
            "summarize this", "summarize file", "tldr", "tl;dr"
        )
        if (exactGlobalPhrases.any { clean.contains(it) }) return true

        val words = clean.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size > 5) return false

        val summaryWords = setOf(
            "summarize", "summary", "overview", "tldr", "tl;dr",
            "about", "doc", "document", "file", "this", "all", "whole",
            "what", "is", "are", "the", "a", "an", "give", "me"
        )
        val nonSummaryWords = words.filter { it !in summaryWords }

        // Only true if no specific topical nouns were introduced
        return nonSummaryWords.isEmpty()
    }

    /**
     * Grounded Multiplicative Section Affinity Calculator.
     * Computes SectionAffinity in [0.0, 1.0] based on the Document Profile's section map.
     */
    fun computeSectionAffinity(
        intent: QueryIntent,
        pageNumber: Int,
        sections: List<DocumentSection>
    ): Double {
        if (sections.isEmpty()) return 0.0

        val activeSection = sections.find { pageNumber in it.pageStart..it.pageEnd } ?: return 0.0
        val titleTokens = activeSection.title.lowercase().split(Regex("[^a-z0-9]+")).toSet()

        return when (intent) {
            is QueryIntent.Definitional -> {
                val definitionalKeywords = setOf("introduction", "overview", "definitions", "principles", "basics", "background")
                if (titleTokens.any { it in definitionalKeywords }) 1.0 else 0.0
            }
            is QueryIntent.Procedural -> {
                val proceduralKeywords = setOf("techniques", "methods", "procedures", "steps", "guide", "checklist", "troubleshooting", "selection")
                if (titleTokens.any { it in proceduralKeywords }) 1.0 else 0.0
            }
            else -> 0.0
        }
    }
}
