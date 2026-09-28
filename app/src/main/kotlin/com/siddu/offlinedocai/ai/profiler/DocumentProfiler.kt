package com.siddu.offlinedocai.ai.profiler

import com.siddu.offlinedocai.domain.rag.ExtractedPage
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min

/**
 * Section span within a document.
 */
data class DocumentSection(
    val pageStart: Int,
    val pageEnd: Int,
    val title: String
)

/**
 * Structured Knowledge Profile for a document.
 */
data class DocumentProfile(
    val schemaVersion: Int = 1,
    val documentId: Long,
    val title: String,
    val domain: String,
    val pageCount: Int,
    val sections: List<DocumentSection>,
    val coreGlossary: List<String>,
    val suggestedQuestions: List<String>
)

/**
 * Deterministic Heuristic Document Profiler (Path A).
 * Executes in < 150ms during document ingestion without heavy neural overhead.
 */
object DocumentProfiler {

    private val STOPWORDS = setOf(
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
        "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    )

    private val NUMBERED_HEADING_REGEX = Regex("""^(?:[0-9]{1,2}\.|\b[IVXLCDM]+\.)\s+([A-Z0-9][^\n.]{3,60})""", RegexOption.MULTILINE)
    private val CAPITALIZED_HEADING_REGEX = Regex("""^[A-Z0-9\s—–-]{4,50}$""", RegexOption.MULTILINE)

    /**
     * Builds a DocumentProfile from extracted pages.
     */
    fun profileDocument(
        documentId: Long,
        title: String,
        pages: List<ExtractedPage>
    ): DocumentProfile {
        val totalPages = pages.size

        // 1. Detect Headings across pages
        val detectedHeadings = mutableListOf<Pair<Int, String>>()
        for (page in pages) {
            val pageLines = page.text.lines().map { it.trim() }.filter { it.length in 4..70 }
            for (line in pageLines) {
                val match = NUMBERED_HEADING_REGEX.find(line)
                if (match != null) {
                    val heading = match.value.trim()
                    if (detectedHeadings.none { it.second.equals(heading, ignoreCase = true) }) {
                        detectedHeadings.add(Pair(page.pageNumber, heading))
                    }
                } else if (CAPITALIZED_HEADING_REGEX.matches(line) && line.split(" ").size in 2..6) {
                    if (detectedHeadings.none { it.second.equals(line, ignoreCase = true) }) {
                        detectedHeadings.add(Pair(page.pageNumber, line))
                    }
                }
            }
        }

        // 2. Build Section Boundaries
        val sections = mutableListOf<DocumentSection>()
        if (detectedHeadings.size >= 2) {
            for (i in detectedHeadings.indices) {
                val current = detectedHeadings[i]
                val nextStart = if (i + 1 < detectedHeadings.size) detectedHeadings[i + 1].first else (totalPages + 1)
                val pageStart = current.first
                val pageEnd = max(pageStart, min(totalPages, nextStart - 1))
                sections.add(DocumentSection(pageStart = pageStart, pageEnd = pageEnd, title = current.second))
            }
        } else {
            // Default 3-tier section layout fallback
            val p1End = max(1, min(2, totalPages))
            val p2End = max(p1End, min(totalPages - 1, totalPages / 2 + 1))
            sections.add(DocumentSection(1, p1End, "Introduction & Overview"))
            if (totalPages > 2) {
                sections.add(DocumentSection(p1End + 1, p2End, "Main Content & Core Concepts"))
                sections.add(DocumentSection(p2End + 1, totalPages, "Detailed Techniques & Summary"))
            }
        }

        // 3. Glossary & Technical Keywords using RAKE over first 3 pages
        val introText = pages.take(3).joinToString(" ") { it.text }
        val glossary = extractRakeKeywords(introText).take(6)

        // 4. Infer Domain
        val domain = inferDomain(title, introText, glossary)

        // 5. Generate 4 Suggested Questions
        val suggestedQuestions = generateSuggestedQuestions(title, glossary, sections)

        return DocumentProfile(
            schemaVersion = 1,
            documentId = documentId,
            title = title,
            domain = domain,
            pageCount = totalPages,
            sections = sections,
            coreGlossary = glossary,
            suggestedQuestions = suggestedQuestions
        )
    }

    /**
     * RAKE (Rapid Automatic Keyword Extraction) algorithm
     */
    private fun extractRakeKeywords(text: String): List<String> {
        val words = text.lowercase().split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotBlank() }
        if (words.isEmpty()) return emptyList()

        // Split text by stopwords and punctuation into candidate phrases
        val phrases = mutableListOf<List<String>>()
        var currentPhrase = mutableListOf<String>()

        for (word in words) {
            if (word in STOPWORDS || word.length <= 2) {
                if (currentPhrase.isNotEmpty()) {
                    phrases.add(currentPhrase)
                    currentPhrase = mutableListOf()
                }
            } else {
                currentPhrase.add(word)
            }
        }
        if (currentPhrase.isNotEmpty()) {
            phrases.add(currentPhrase)
        }

        // Compute word frequency and degree
        val wordFreq = mutableMapOf<String, Int>()
        val wordDegree = mutableMapOf<String, Int>()

        for (phrase in phrases) {
            val degree = phrase.size
            for (w in phrase) {
                wordFreq[w] = (wordFreq[w] ?: 0) + 1
                wordDegree[w] = (wordDegree[w] ?: 0) + degree
            }
        }

        val wordScores = mutableMapOf<String, Double>()
        for ((w, freq) in wordFreq) {
            val deg = wordDegree[w] ?: freq
            wordScores[w] = deg.toDouble() / freq.toDouble()
        }

        // Score phrases
        val phraseScores = mutableMapOf<String, Double>()
        for (phrase in phrases) {
            if (phrase.size in 1..4) {
                val phraseStr = phrase.joinToString(" ")
                if (phraseStr.length >= 4) {
                    var score = 0.0
                    for (w in phrase) {
                        score += wordScores[w] ?: 1.0
                    }
                    phraseScores[phraseStr] = score
                }
            }
        }

        return phraseScores.entries
            .sortedByDescending { it.value }
            .map { it.key }
            .distinct()
    }

    private fun inferDomain(title: String, intro: String, glossary: List<String>): String {
        val combined = "$title $intro ${glossary.joinToString(" ")}".lowercase()
        return when {
            listOf("plant", "graft", "horticulture", "agriculture", "crop", "botany").any { combined.contains(it) } -> "Plant Propagation / Horticulture"
            listOf("software", "api", "code", "architecture", "database", "rag").any { combined.contains(it) } -> "Computer Science / Software Engineering"
            listOf("law", "legal", "clause", "contract", "agreement", "party").any { combined.contains(it) } -> "Legal Document / Contract"
            listOf("medical", "patient", "clinical", "health", "aid", "treatment").any { combined.contains(it) } -> "Medical / Health Sciences"
            listOf("finance", "revenue", "fiscal", "tax", "accounting").any { combined.contains(it) } -> "Finance / Economics"
            else -> "General Domain / Technical Document"
        }
    }

    private fun generateSuggestedQuestions(
        title: String,
        glossary: List<String>,
        sections: List<DocumentSection>
    ): List<String> {
        val primaryTerm = glossary.firstOrNull()?.replaceFirstChar { it.uppercase() } ?: "the main topic"
        val secondaryTerm = glossary.getOrNull(1)?.replaceFirstChar { it.uppercase() } ?: "key principles"

        val q1 = "What is the basic definition and purpose of $primaryTerm?"
        val q2 = "How does $secondaryTerm relate to the core concepts in this document?"

        val techSection = sections.find { sec ->
            listOf("technique", "method", "procedure", "step", "guide").any { sec.title.lowercase().contains(it) }
        }
        val q3 = if (techSection != null) {
            "What are the main procedures or techniques outlined in '${techSection.title}'?"
        } else {
            "What are the key practical recommendations described in this document?"
        }

        val q4 = "What are the common challenges, checklists, or failure causes discussed?"

        return listOf(q1, q2, q3, q4)
    }

    /**
     * Serializes profile into JSON string.
     */
    fun toJson(profile: DocumentProfile): String {
        val obj = JSONObject()
        obj.put("schemaVersion", profile.schemaVersion)
        obj.put("documentId", profile.documentId)
        obj.put("title", profile.title)
        obj.put("domain", profile.domain)
        obj.put("pageCount", profile.pageCount)

        val sectionsArr = JSONArray()
        for (sec in profile.sections) {
            val secObj = JSONObject()
            secObj.put("pageStart", sec.pageStart)
            secObj.put("pageEnd", sec.pageEnd)
            secObj.put("title", sec.title)
            sectionsArr.put(secObj)
        }
        obj.put("sections", sectionsArr)

        val glossaryArr = JSONArray()
        profile.coreGlossary.forEach { glossaryArr.put(it) }
        obj.put("coreGlossary", glossaryArr)

        val questionsArr = JSONArray()
        profile.suggestedQuestions.forEach { questionsArr.put(it) }
        obj.put("suggestedQuestions", questionsArr)

        return obj.toString()
    }

    /**
     * Deserializes profile from JSON string.
     */
    fun fromJson(jsonStr: String): DocumentProfile? {
        return try {
            val obj = JSONObject(jsonStr)
            val docId = obj.optLong("documentId", 0L)
            val title = obj.optString("title", "")
            val domain = obj.optString("domain", "")
            val pageCount = obj.optInt("pageCount", 0)

            val sections = mutableListOf<DocumentSection>()
            val secArr = obj.optJSONArray("sections")
            if (secArr != null) {
                for (i in 0 until secArr.length()) {
                    val sObj = secArr.getJSONObject(i)
                    sections.add(
                        DocumentSection(
                            pageStart = sObj.optInt("pageStart", 1),
                            pageEnd = sObj.optInt("pageEnd", 1),
                            title = sObj.optString("title", "")
                        )
                    )
                }
            }

            val glossary = mutableListOf<String>()
            val gArr = obj.optJSONArray("coreGlossary")
            if (gArr != null) {
                for (i in 0 until gArr.length()) {
                    glossary.add(gArr.getString(i))
                }
            }

            val questions = mutableListOf<String>()
            val qArr = obj.optJSONArray("suggestedQuestions")
            if (qArr != null) {
                for (i in 0 until qArr.length()) {
                    questions.add(qArr.getString(i))
                }
            }

            DocumentProfile(
                schemaVersion = obj.optInt("schemaVersion", 1),
                documentId = docId,
                title = title,
                domain = domain,
                pageCount = pageCount,
                sections = sections,
                coreGlossary = glossary,
                suggestedQuestions = questions
            )
        } catch (e: Exception) {
            null
        }
    }
}
