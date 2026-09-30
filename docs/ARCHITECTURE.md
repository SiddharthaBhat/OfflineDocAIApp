# OfflineDocAI — Complete Technical Architecture

> **Project:** OfflineDocAI — Offline Mobile Retrieval-Augmented Generation (RAG) App  
> **Platform:** Android (minSdk 26 / targetSdk 34)  
> **Language:** Kotlin + Java (JNI bridge), Jetpack Compose UI  
> **Last Updated:** September 2026 (State 004 — Post Bug-Fix Stabilization)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Deep-Dive](#4-component-deep-dive)
   - 4.1 [Document Ingestion Pipeline](#41-document-ingestion-pipeline)
   - 4.2 [Embedding Engine](#42-embedding-engine-ggufembeddingengine)
   - 4.3 [Vector Store & Hybrid Retrieval](#43-vector-store--hybrid-retrieval)
   - 4.4 [Query Reformulator](#44-query-reformulator)
   - 4.5 [Selective Content Reducer (SCR)](#45-selective-content-reducer-scr)
   - 4.6 [Memory Governor](#46-memory-governor)
   - 4.7 [On-Device LLM (GGUFLLM)](#47-on-device-llm-ggufllm)
   - 4.8 [Document Profiler](#48-document-profiler)
   - 4.9 [Chat Repository (RAG Orchestrator)](#49-chat-repository-rag-orchestrator)
   - 4.10 [Room Database Schema](#410-room-database-schema)
   - 4.11 [Native JNI Bridge (LlamaContext)](#411-native-jni-bridge-llamacontext)
5. [End-to-End Data Flows](#5-end-to-end-data-flows)
   - 5.1 [Document Import Flow](#51-document-import-flow)
   - 5.2 [RAG Chat Query Flow](#52-rag-chat-query-flow)
   - 5.3 [General AI Chat Flow](#53-general-ai-chat-flow)
6. [Supported AI Models](#6-supported-ai-models)
7. [Memory & Performance Architecture](#7-memory--performance-architecture)
8. [Package Structure](#8-package-structure)
9. [Key Design Decisions & Trade-offs](#9-key-design-decisions--trade-offs)
10. [Known Limitations](#10-known-limitations)

---

## 1. System Overview

OfflineDocAI is a **fully offline, on-device Retrieval-Augmented Generation (RAG) application** for Android. It allows users to:

1. **Import documents** (PDF, DOCX, TXT) and have them automatically chunked, embedded, and indexed in a local Room database.
2. **Ask natural-language questions** about any imported document — the system retrieves the most relevant passages and uses an on-device Small Language Model (SLM) to synthesize a grounded answer.
3. **Use the SLM as a general-purpose chat assistant** (when no document is selected) for factual questions, math, and reasoning tasks.

**No data ever leaves the device.** All inference — embedding generation and text generation — runs locally using `librnllama.so` (llama.cpp compiled for Android ARMv8 / x86_64).

---

## 2. Technology Stack

| Layer | Technology | Version / Detail |
|---|---|---|
| **UI** | Jetpack Compose + Material 3 | Compose 1.7.2, Material3 1.3.0 |
| **Navigation** | Navigation Compose | 2.8.1 |
| **ViewModel** | AndroidViewModel + StateFlow | lifecycle-runtime-ktx 2.8.2 |
| **Local Database** | Room (SQLite) | 2.6.1 |
| **PDF Parsing** | PdfBox-Android (Tom Roush) | 2.0.27.0 |
| **DOCX Parsing** | Apache POI OOXML | 5.2.3 |
| **LLM Inference** | llama.cpp via JNI (`librnllama.so`) | Custom Android JNI bridge (`LlamaContext.java`) |
| **Embedding Inference** | llama.cpp (`LlamaContext`, `embedding=true`) | Same `.so` as LLM, pooled mode |
| **ONNX Fallback** | ONNX Runtime Android | 1.18.0 (dependency present, not yet used for main path) |
| **Concurrency** | Kotlin Coroutines | 1.8.0 |
| **Build** | Gradle Kotlin DSL | compileSdk 34, minSdk 26 |
| **Language** | Kotlin 2.0+ (with Java interop for JNI) | kapt for Room annotation processing |

### Native Libraries (JNI `.so` files)

Located in `app/src/main/jniLibs/`:

| ABI | Libraries |
|---|---|
| `arm64-v8a` | `librnllama.so`, `librnllama_v8.so`, `librnllama_v8_2.so`, `librnllama_v8_2_dotprod.so`, `librnllama_v8_2_dotprod_i8mm.so`, `librnllama_v8_2_i8mm.so` |
| `x86_64` | `librnllama.so`, `librnllama_x86_64.so` |

The `LlamaContext` static initializer selects the best `.so` variant at runtime by reading `/proc/cpuinfo` for ARM feature flags (`dotprod`, `i8mm`, `asimd`).

---

## 3. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         OFFLINE DOC AI — SYSTEM MAP                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  ANDROID UI LAYER (Jetpack Compose)                                         │
│  MainActivity → MainViewModel → Navigation                                  │
│  • Document List Screen    • Chat Screen    • Model Manager Screen           │
└─────────────────────┬───────────────────────────────┬───────────────────────┘
                      │ importDocument(uri)            │ sendMessage(convId, text, docIds)
                      ▼                               ▼
┌─────────────────────────────┐     ┌────────────────────────────────────────┐
│  DocumentRepository         │     │  ChatRepository (RAG Orchestrator)     │
│  • PDF/DOCX/TXT extraction  │     │  • Query reformulation                 │
│  • Sentence-aware chunking  │     │  • Embedding query                     │
│  • Embedding generation     │     │  • Hybrid retrieval                    │
│  • Room persistence         │     │  • SCR compression                     │
│  • Document Profiling       │     │  • LLM generation                      │
└────────────┬────────────────┘     └────────────────┬───────────────────────┘
             │                                       │
             ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROOM DATABASE  (offline_docai.db)                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  documents  │  │   chunks    │  │conversations │  │    messages      │ │
│  │  (metadata  │  │  (text +    │  │  (chat       │  │  (history +      │ │
│  │  + profile) │  │  embedding) │  │  threads)    │  │  sources JSON)   │ │
│  └─────────────┘  └─────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
             │                                       │
             ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ON-DEVICE AI LAYER                                                         │
│  ┌────────────────────────────────┐   ┌───────────────────────────────────┐ │
│  │  GGUFEmbeddingEngine           │   │  GGUFLLM                          │ │
│  │  LlamaContext(embedding=true)  │   │  LlamaContext(embedding=false)    │ │
│  │  → real pooled embeddings OR   │   │  → real autoregressive generation │ │
│  │  → 768-dim hash fallback       │   │  → intelligent extractive fallback│ │
│  └────────────────────────────────┘   └───────────────────────────────────┘ │
│                          ▲                         ▲                        │
│                          └─────────────────────────┘                       │
│                              LlamaContext.java (JNI)                        │
│                              ↓ librnllama.so (llama.cpp)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Deep-Dive

### 4.1 Document Ingestion Pipeline

**Files:** `DocumentProcessor.kt`, `DocumentRepository.kt`, `TextChunker.kt`

The ingestion pipeline transforms a raw document URI into a set of indexed, embedded chunks stored in Room. It runs as a coroutine on `Dispatchers.IO`.

#### Stages

```
URI
 │
 ▼ ① Format Detection (MIME type or file extension)
DocumentProcessor.extractDocument()
 │
 │  PDF  → PDFBox-Android PDFTextStripper (page-by-page)
 │  DOCX → Apache POI XWPFDocument (whole-doc, single page)
 │  TXT  → BufferedReader (whole-doc, single page)
 │
 ▼ ② Boilerplate Stripping (cleanBoilerplate)
   • Strips standalone page numbers (e.g. "12", "Page 12 of 15")
   • Strips decorative lines (----, ====, ~~~~)
   • Collapses 3+ consecutive blank lines to 2
 │
 ▼ ③ ExtractedDocument { pageCount, List<ExtractedPage> }
```

#### Chunking — TextChunker

**File:** `TextChunker.kt`

Uses a **sentence-aware sliding window** across all pages:

1. Every page's text is split into sentences using `BreakIterator.getSentenceInstance(Locale.US)` — respects abbreviations (Dr., e.g., vs.) by merging incorrectly split fragments.
2. A sliding window accumulates sentences until the word budget (`targetWords = 300`) is met.
3. A backward overlap (`overlapWords = 60`) rewinds to retain context continuity between consecutive chunks.
4. **Cross-page continuity**: sentences flow across page boundaries without hard breaks — a chunk spanning pages 2-3 gets its primary page attribution from its central sentence.

```
targetWords = 300 words per chunk
overlapWords = 60 words of backward overlap
Primary page = page of the median sentence in the chunk
```

#### Embedding

After chunking, `GGUFEmbeddingEngine.getEmbedding(chunk.text)` is called for each chunk sequentially. The resulting `FloatArray` is serialized to `ByteArray` via `FloatArrayConverter` (4 bytes per float, big-endian `ByteBuffer`) and stored as `chunks.embedding`.

#### Document Profile

After all chunks are embedded and saved, `DocumentProfiler.profileDocument()` runs **once** at ingestion end. It produces a `DocumentProfile` JSON blob stored in `documents.profileJson`. See [§4.8](#48-document-profiler) for details.

#### Incremental Ingestion Cache

Before any extraction begins, `documentDao.getByUri(uri)` checks if a `COMPLETED` document with the same URI already exists. If yes → ingestion is skipped entirely (RAGdb-style O(U) vs O(N) savings).

---

### 4.2 Embedding Engine (GGUFEmbeddingEngine)

**File:** `GGUFEmbeddingEngine.kt`  
**Interface:** `EmbeddingModel`

```kotlin
interface EmbeddingModel {
    suspend fun getEmbedding(text: String): FloatArray
}
```

**Dual-mode operation:**

#### Mode 1 — Native GGUF Neural Embedding (preferred)

When a GGUF embedding model (e.g., `all-MiniLM-L6-v2.Q4_K_M.gguf`) is present in `filesDir/models/embeddings/`:

1. `LlamaContext(id=3, modelFile, nCtx=512, nBatch=512, nThreads=4, embedding=true)` is constructed at init.
2. `getEmbedding()` calls `ctx.getEmbeddingVector(text)` → native JNI → llama.cpp pooled mean-pooling → dense float vector.
3. The returned `FloatArray` is L2-normalized before storage.
4. The embedding dimension matches the model (e.g., 384 for all-MiniLM-L6-v2).

#### Mode 2 — 768-dim Hash Fallback (no model / native failure)

When no model is present or native loading fails:

```
tokens = text.lowercase().split(non-alphanumeric).filter { len > 1 }

For each (term, count):
  weight = 1 + ln(count)            // sublinear TF
  vector[hash(term) % 768]       += weight
  vector[hash(term+"§") % 768]   += weight * 0.6
  vector[hash(term+len) % 768]   += weight * 0.3

For each bigram (t[i], t[i+1]):
  vector[hash(bigram) % 768]      += 1.5
  vector[hash(bigram+"_b") % 768] += 0.75

For each trigram (t[i], t[i+1], t[i+2]):
  vector[hash(trigram) % 768]     += 0.8

→ L2 normalize
```

768 dimensions (doubled from the original broken 384-dim) with unigram+bigram+trigram hashing significantly reduces hash collision impact and captures phrase-level semantics.

> **Important:** Existing documents embedded with the old 384-dim engine have incompatible stored embeddings. The `cosineSimilarity()` function in VectorStore returns `0f` when dimensions differ, so old chunks fall back to lexical-only scoring. Re-importing a document generates correct new embeddings automatically.

---

### 4.3 Vector Store & Hybrid Retrieval

**File:** `VectorStore.kt`

Implements **Two-Stage Hybrid Retrieval** based on Pocket RAG §3.3 and RAGdb Hybrid Scoring Function (HSF).

#### Stage 1 — Lexical Pre-filter

Computes a fast lexical relevance score for every chunk in the database for the target documents:

```
Keywords K_q = tokenize(query) \ stopwords

For each chunk c_i:
  For each keyword kw ∈ K_q:
    occurrences  = count(kw, chunk)
    sublinearTF  = 1 + ln(occurrences)
    idf          = ln((N+1) / (df+0.5)) + 1
    sLex        += sublinearTF × idf

  keywordCoverage = |K_q ∩ W_c| / |K_q|   (fraction of keywords hit)
  phraseBoost = 15.0 if exact phrase present
  totalLexScore = (keywordCoverage × 20) + sLex + phraseBoost
```

All chunks are sorted by `totalLexScore`, normalized to [0, 1], and the top `max(topK×3, 40)` are retained as the candidate pool.

#### Stage 2 — Semantic Re-ranking

For each candidate:
```
cosineScore = cosineSimilarity(E(query), E(chunk))   // 0 if dim mismatch
baseScore   = 0.60 × cosineScore + 0.40 × sLex_norm
sectionAffinity = QueryReformulator.computeSectionAffinity(intent, pageNum, profile)
positionalBoost = {0.60, 0.40, 0.20, 0} for pages {1, 2, 3, other} (Definitional only)

finalScore = baseScore × (1 + 0.40 × sectionAffinity + positionalBoost)
```

Alpha = 0.60 (semantic weight) is the Pocket RAG optimal value.

#### TopK Selection (Intent-Aware)

```kotlin
effectiveTopK = when (intent) {
    Definitional  → 2           // high precision, early pages
    GlobalSummary → max(topK, 3)// breadth for overview
    Procedural    → topK in 3..5// moderate breadth
    DomainSpecific→ topK in 3..5// full governor allocation
}
```

`topK` comes from `MemoryGovernor.getAdaptiveConfig()` (5 / 3 / 2 depending on memory pressure).

#### Special Modes

- **PageNavigation intent**: bypasses all scoring, directly queries `chunkDao.getChunksForDocumentAndPage()`.
- **GlobalSummary intent**: calls `selectStructuralOverviewChunks()` — samples preamble (first chunk), evenly spaced interior chunks, and conclusion chunk for document-wide coverage.

---

### 4.4 Query Reformulator

**File:** `QueryReformulator.kt`

Performs **deterministic intent classification** and search string rewriting without any neural inference.

#### Intent Taxonomy

| Intent Class | Trigger | Example | Search Rewrite |
|---|---|---|---|
| `PageNavigation` | "page N", "go to page 3" | "show me page 5" | raw query |
| `GlobalSummary` | strict dominance check on summary vocab | "summarize this document" | raw query |
| `Definitional` | "what is X", "define X", "explain X", single noun | "what is grafting" | "grafting definition" |
| `Procedural` | "how to X", "steps for X", "why is X" | "how to select a plant for grafting" | "select a plant for grafting procedure guide" |
| `DomainSpecific` | fallback | "which rootstock is best for apples" | raw query |

#### Section Affinity

`computeSectionAffinity()` maps `(intent, pageNumber, documentSections)` → `[0, 1]`:
- `Definitional` intent → affinity=1 for sections with titles containing `introduction/overview/definitions/principles`
- `Procedural` intent → affinity=1 for sections with `techniques/methods/procedures/steps/guide/checklist`

This affinity multiplicatively boosts chunk scores during re-ranking, anchoring retrieval to the document region most likely to contain the right type of answer.

---

### 4.5 Selective Content Reducer (SCR)

**File:** `SelectiveContentReducer.kt`

After retrieval, each chunk's full text (~300 words) is compressed before inclusion in the LLM prompt. This directly reduces **Time-to-First-Token (TTFT)** and prevents context window overflow in small SLMs.

#### Algorithm

```
For each retrieved chunk:
  sentences = splitIntoSentences(chunk.text)   // regex (?<=[.!?])\s+
  
  For each sentence s:
    score  = (keyword_hits / |K_q|) × 10       // coverage
    score += 15 if exact phrase present         // phrase bonus
    score += 2 if index == 0                    // first sentence bonus
    score += 1 if index == last                 // last sentence bonus
    score += 1.5 if sentence contains digits    // data density bonus
  
  retain top (retentionRatio × |sentences|) sentences by score
  restore original order (sorted by index)
  → compressedText

tokenReductionPercentage = 1 - (len(compressed) / len(original))
```

**Target retention ratios** (from MemoryGovernor):
- SAFE: 0.75 (retain 75% of sentences)
- MODERATE: 0.60
- CRITICAL: 0.40

For `isGenericQuery=true` (overview/summary queries), SCR takes leading sentences without keyword filtering to preserve introductory context.

#### Context Budget

`formatCompressedContext()` enforces a hard **2400-character context budget** across all chunks. If adding the next chunk would exceed the budget, only a sentence-boundary truncated fragment is appended.

---

### 4.6 Memory Governor

**File:** `MemoryGovernor.kt`

Enforces the Pocket RAG / HeRo memory constraint: `M_total ≤ 2.0 GB`.

```kotlin
pressureRatio ρ = (totalMemory - freeMemory) / maxMemory

ρ < 0.70  → SAFE    : topK=5, maxContextTokens=1024, scrRetentionRatio=0.75
ρ < 0.85  → MODERATE: topK=3, maxContextTokens=768,  scrRetentionRatio=0.60
ρ ≥ 0.85  → CRITICAL: topK=2, maxContextTokens=384,  scrRetentionRatio=0.40
```

Called once per query in `ChatRepository.executeMessage()` before retrieval. The returned `GovernanceConfig` feeds into VectorStore and SCR.

---

### 4.7 On-Device LLM (GGUFLLM)

**File:** `GGUFLLM.kt`  
**Interface:** `LocalLLM`

```kotlin
interface LocalLLM {
    suspend fun generate(prompt: String): String
    suspend fun generate(prompt: String, systemPrompt: String?): String
}
```

#### Native Inference Path

When a GGUF LLM file is present in `filesDir/models/llm/`:

1. `LlamaContext(id=2, modelFile, nCtx=4096, nBatch=512, nThreads=4, embedding=false)` is lazily initialized (once per model file, reused across queries).
2. `formatPromptForLlm()` wraps the prompt+system in model-specific chat tokens:

| Model Family | Chat Format |
|---|---|
| Qwen 2.5 | ChatML (`<\|im_start\|>system ... <\|im_end\|>`) |
| Llama 3.x | Llama 3 (`<\|begin_of_text\|><\|start_header_id\|>...`) |
| DeepSeek | DeepSeek (`<｜begin of sentence｜>...`) |
| Gemma | `<start_of_turn>user ...` |
| Phi | `<\|system\|>...<\|end\|>` |
| Generic | Alpaca (`### Instruction:`) |

3. `generateCompletion()` is called with:
   - RAG mode: `temp=0.25, maxTokens=400` (low temperature for factual grounding)
   - Pure chat mode: `temp=0.60, maxTokens=512`
   - Stop words: `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `</s>`, `\nUser:`, `\nHuman:`

#### Intelligent Extractive Fallback (No LLM model)

When no model is present, `generateFallback()` routes to one of two synthesizers:

**RAG mode — `synthesizeRagAnswer(question, contextText)`:**

```
1. Detect query type: definition / list / procedural / overview / factual
2. Extract (sentence, pageNumber) pairs from context
3. Score each sentence:
   - keyword hits × 3.0
   - keyword coverage × 5.0
   - exact phrase bonus × 10.0
   - definitional phrasing bonus ("is a", "refers to", ...) × 4.0  [if definition query]
   - list item bonus × 3.0  [if list query]
4. Select top-N sentences by type:
   - definition → lead with most definitional sentence + 2 support
   - list → surface bullet/numbered items; format as "• item"
   - procedural → "Steps / Procedure:" heading + top 5 steps
   - overview → top 5 sentences as paragraph
5. Append page attribution: "(Source pages: 2, 3, 5)"
```

**General chat — `synthesizeGeneralAnswer(question)`:**

```
Priority chain:
1. Conversational greetings → direct friendly response
2. Identity/model questions → report actual installed model name or "no model" info
3. Known factual answers → apple colors, speed of light, boiling point, etc.
4. Math evaluation → arithmetic, %, sqrt, powers
5. Anything requiring genuine reasoning → buildModelNeededResponse():
   - Honest "no model installed" message
   - Contextual hint based on question type (doc question / code / explain)
   - Instructions to download a model from the Models tab
```

> No fake templates. No hallucinated boilerplate. Every response is either grounded in context, factually correct, or honestly says "I need a model for this."

---

### 4.8 Document Profiler

**File:** `DocumentProfiler.kt`

Runs **once per document** at ingestion end. Produces a `DocumentProfile` stored as JSON in `documents.profileJson`.

#### DocumentProfile Schema

```json
{
  "schemaVersion": 1,
  "documentId": 1,
  "title": "Grafting_15_Page_Guide.pdf",
  "domain": "Plant Propagation / Horticulture",
  "pageCount": 15,
  "sections": [
    { "pageStart": 2, "pageEnd": 2, "title": "1. What Is Grafting?" },
    { "pageStart": 3, "pageEnd": 3, "title": "2. The Biology Behind a Graft Union" }
  ],
  "coreGlossary": ["cambial contact", "scion rootstock", "vascular tissues", ...],
  "suggestedQuestions": [
    "What is the basic definition and purpose of Cambial contact?",
    "How does Scion rootstock relate to the core concepts?",
    ...
  ]
}
```

#### Algorithms

**Section Detection:**
- Scans every page for numbered headings: `Regex("^(?:[0-9]{1,2}\.|[IVXLCDM]+\.)\s+([A-Z0-9][^.]{3,60})")`
- Also detects ALL-CAPS headings: `Regex("^[A-Z0-9\s—–-]{4,50}$")` with 2-6 word limit
- Builds `DocumentSection(pageStart, pageEnd, title)` spans
- Falls back to 3-tier layout for docs with <2 detected headings

**Glossary Extraction — RAKE:**
- Split text into candidate phrases at stopwords
- Score each word: `score(w) = degree(w) / freq(w)`
- Score each phrase: `sum of word scores`
- Top 6 phrases become `coreGlossary`

**Domain Inference:**
- Keyword matching on title + intro text + glossary
- Categories: Plant Propagation, CS/Software, Legal, Medical, Finance, General

---

### 4.9 Chat Repository (RAG Orchestrator)

**File:** `ChatRepository.kt`

The central coordinator of the RAG pipeline. `executeMessage()` runs on `Dispatchers.IO` and orchestrates the entire query-to-answer flow:

```kotlin
suspend fun executeMessage(conversationId, userText, relevantDocumentIds) {
    // 1. Persist user message to Room
    
    if (cleanDocIds.isEmpty()) {
        // General AI Chat path
        response = llm.generate(userText, normalSystemPrompt)
    } else {
        // RAG path
        profile = documentDao.getById(docId)?.profileJson → DocumentProfiler.fromJson()
        reformulation = QueryReformulator.classifyAndRewrite(userText, profile)
        
        if (GlobalSummary + profile exists) {
            // Short-circuit: return formatted section map from profile (0ms vector search)
        } else {
            govConfig = MemoryGovernor.getAdaptiveConfig()
            queryEmbedding = embeddingEngine.getEmbedding(reformulation.searchString)
            rawChunks = vectorStore.retrieve(queryEmbedding, docIds, govConfig.topK, ...)
            compressedChunks = SCR.reduceChunks(rawChunks, userText, govConfig.scrRetentionRatio)
            contextText = SCR.formatCompressedContext(compressedChunks, maxChars=2400)
            
            ragSystemPrompt = "You are an expert offline document AI assistant..."
            ragPrompt = "Document Context:\n$contextText\n\nQuestion: $userText"
            response = llm.generate(ragPrompt, ragSystemPrompt)
        }
    }
    
    // Simulate token streaming: insert empty message, update word-by-word with 30ms delay
    // Persist final message with sources JSON
}
```

#### Token Streaming Simulation

Since llama.cpp's JNI callback (`onPartialCompletion`) passes tokens as they're generated, the current implementation collects the full response and then re-streams it word-by-word with 30ms delay. This gives a live-typing UX even without true streaming integration.

---

### 4.10 Room Database Schema

**File:** `AppDatabase.kt` (version 3)

```
┌──────────────────────────────────────────────────────────┐
│ TABLE: documents                                         │
├──────────────┬────────────────────────────────────────── │
│ id           │ Long (PK autoGenerate)                    │
│ name         │ String (filename)                         │
│ uri          │ String (content URI)                      │
│ type         │ String (MIME type)                        │
│ pageCount    │ Int                                       │
│ addedAt      │ Long (epoch ms)                           │
│ status       │ String (PROCESSING / COMPLETED / FAILED)  │
│ profileJson  │ String? (DocumentProfile JSON blob)       │
└──────────────┴────────────────────────────────────────── ┘

┌──────────────────────────────────────────────────────────┐
│ TABLE: chunks                                            │
├──────────────┬────────────────────────────────────────── │
│ id           │ Long (PK autoGenerate)                    │
│ documentId   │ Long (FK → documents.id)                  │
│ pageNumber   │ Int                                       │
│ text         │ String (~300 words per chunk)             │
│ embedding    │ ByteArray? (FloatArray as 4×N bytes)      │
└──────────────┴────────────────────────────────────────── ┘

┌──────────────────────────────────────────────────────────┐
│ TABLE: conversations                                     │
├──────────────┬────────────────────────────────────────── │
│ id           │ Long (PK autoGenerate)                    │
│ title        │ String                                    │
│ createdAt    │ Long (epoch ms)                           │
│ documentId   │ Long? (null = general AI chat)            │
└──────────────┴────────────────────────────────────────── ┘

┌──────────────────────────────────────────────────────────┐
│ TABLE: messages                                          │
├──────────────┬────────────────────────────────────────── │
│ id           │ Long (PK autoGenerate)                    │
│ conversationId│ Long (FK → conversations.id)             │
│ role         │ String ("user" / "assistant")             │
│ content      │ String (message text, streamed live)      │
│ timestamp    │ Long (epoch ms)                           │
│ sources      │ String? ("Page N: text|||Page M: text")   │
│ thinkingDurationMs│ Long? (ms from send to first token)  │
└──────────────┴────────────────────────────────────────── ┘
```

**Type Converters:**
- `FloatArrayConverter`: `FloatArray ↔ ByteArray` via `ByteBuffer.putFloat` (4 bytes/float)
- `Converters`: handles any additional type conversions (timestamps, etc.)

**Migrations:**
- Migration 2→3: `ALTER TABLE messages ADD COLUMN thinkingDurationMs INTEGER DEFAULT NULL`

---

### 4.11 Native JNI Bridge (LlamaContext)

**File:** `LlamaContext.java`

Java class that wraps the llama.cpp native library. Two key uses:

#### Text Generation (LLM)
```java
LlamaContext ctx = new LlamaContext(id=2, modelFile, nCtx=4096, nBatch=512, nThreads=4, embedding=false);
String result = ctx.generateCompletion(prompt, temperature, topK, topP, maxTokens, stopWords, callback);
```

The `initContextWithFd()` native call uses `ParcelFileDescriptor.open()` + `pfd.detachFd()` to safely hand the file descriptor to native C++ without triggering Android's `fdsan` SIGABRT (which occurs when native `close(fd)` is called on a Java-owned descriptor).

#### Embedding Generation
```java
LlamaContext ctx = new LlamaContext(id=3, embeddingModel, nCtx=512, nBatch=512, nThreads=4, embedding=true);
float[] vector = ctx.getEmbeddingVector(text);
```

The `embedding=true` flag tells llama.cpp to enable pooled embedding mode, returning the mean-pooled final-layer activations instead of generating new tokens.

#### CPU Feature Detection

At class load time, the static initializer reads `/proc/cpuinfo` and loads the optimal `.so`:

```
arm64 + v8.2 + dotprod + i8mm → librnllama_v8_2_dotprod_i8mm.so  (fastest)
arm64 + v8.2 + dotprod        → librnllama_v8_2_dotprod.so
arm64 + v8.2 + i8mm           → librnllama_v8_2_i8mm.so
arm64 + v8.2                  → librnllama_v8_2.so
arm64                         → librnllama_v8.so
x86_64                        → librnllama_x86_64.so
fallback                      → librnllama.so
```

---

## 5. End-to-End Data Flows

### 5.1 Document Import Flow

```
User picks file (SAF picker)
        │
        ▼
DocumentRepository.importDocument(uri)   [Dispatchers.IO]
        │
        ├─ Check: documentDao.getByUri(uri) already COMPLETED? → skip (incremental cache)
        │
        ├─ DocumentProcessor.extractDocument(context, uri)
        │     ├─ PDF: PDFBox-Android, page-by-page text extraction
        │     ├─ DOCX: Apache POI, paragraph text join
        │     └─ TXT: BufferedReader line join
        │     └─ cleanBoilerplate(): strip page nums, decorators, collapse whitespace
        │     → ExtractedDocument { pageCount, List<ExtractedPage(pageNum, text)> }
        │
        ├─ documentDao.insert(DocumentEntity(status=PROCESSING))  → docId
        │
        ├─ TextChunker.chunkDocument(pages, targetWords=300, overlapWords=60)
        │     → List<Chunk { index, pageNumber, text }>
        │     (sentence-aware, cross-page, sliding window)
        │
        ├─ For each chunk:
        │     embedding = GGUFEmbeddingEngine.getEmbedding(chunk.text)
        │     bytes = FloatArrayConverter.fromFloatArray(embedding)
        │     allChunks += ChunkEntity(docId, pageNum, text, bytes)
        │
        ├─ chunkDao.insertAll(allChunks)
        │
        ├─ DocumentProfiler.profileDocument(docId, fileName, pages)
        │     → DocumentProfile { domain, sections, glossary, questions }
        │     → profileJson = DocumentProfiler.toJson(profile)
        │
        └─ documentDao.update(status=COMPLETED, profileJson=profileJson)

importState: Idle → Reading → Chunking → Embedding → Saving → Completed
```

### 5.2 RAG Chat Query Flow

```
User types question in Chat screen
        │
        ▼
ChatRepository.sendMessage(convId, userText, [docId])
        │
        ▼
executeMessage() [Dispatchers.IO]
        │
        ├─ messageDao.insert(MessageEntity(role="user", content=userText))
        │
        ├─ primaryDoc = documentDao.getById(docId)
        │   profile = DocumentProfiler.fromJson(primaryDoc.profileJson)
        │
        ├─ reformulation = QueryReformulator.classifyAndRewrite(userText, profile)
        │   → intent: PageNavigation / GlobalSummary / Definitional / Procedural / DomainSpecific
        │   → searchString: rewritten for retrieval
        │
        ├─ [GlobalSummary short-circuit]
        │   If intent=GlobalSummary AND profile!=null:
        │     → return formatted section map from profile (0ms, no vector search)
        │
        ├─ govConfig = MemoryGovernor.getAdaptiveConfig()   // topK, scrRatio
        │
        ├─ queryEmbedding = GGUFEmbeddingEngine.getEmbedding(reformulation.searchString)
        │   (native GGUF OR 768-dim hash fallback)
        │
        ├─ rawChunks = VectorStore.retrieve(
        │       queryEmbedding, [docId], govConfig.topK,
        │       query=reformulation.searchString,
        │       intent=reformulation.intent,
        │       profile=profile)
        │   Stage 1: Lexical TF-IDF pre-filter → candidate pool of 40
        │   Stage 2: Hybrid cosine+lexical scoring → sorted → topK (intent-aware 2-5)
        │
        ├─ compressedChunks = SCR.reduceChunks(rawChunks, userText, govConfig.scrRetentionRatio)
        │   (sentence scoring, retention, order preservation)
        │
        ├─ contextText = SCR.formatCompressedContext(compressedChunks, maxChars=2400)
        │   "[Page 2] sentence1 sentence2... \n\n[Page 5] sentence3..."
        │
        ├─ ragPrompt = "Document Context:\n{contextText}\n\nQuestion: {userText}"
        │   ragSystemPrompt = "You are an expert offline document AI assistant..."
        │
        ├─ response = GGUFLLM.generate(ragPrompt, ragSystemPrompt)
        │   → LlamaContext.generateCompletion() if model present
        │   → synthesizeRagAnswer() if no model (extractive NL synthesis)
        │
        ├─ sourcesJson = "Page N: {compressedText}|||Page M: {compressedText}"
        │
        └─ messageDao.insert(empty) → stream word-by-word (30ms/word) → messageDao.update()
           (with sources JSON for expandable source attribution UI)
```

### 5.3 General AI Chat Flow

```
User types in General Chat (no document linked)
        │
        ▼
ChatRepository.executeMessage(convId, userText, docIds=[])
        │
        ├─ cleanDocIds.isEmpty() → true → skip RAG path
        │
        ├─ normalSystemPrompt = "You are an intelligent, versatile offline AI assistant..."
        ├─ response = GGUFLLM.generate(userText, normalSystemPrompt)
        │
        │   ── if LLM model present ──
        │   formatPromptForLlm(modelName, userText, normalSystemPrompt)
        │   → LlamaContext.generateCompletion(formattedPrompt, temp=0.6, maxTokens=512)
        │   → real on-device autoregressive generation
        │
        │   ── if no LLM model ──
        │   generateFallback(prompt=userText, systemPrompt=normalSystemPrompt)
        │   → synthesizeGeneralAnswer(userText)
        │       ├─ greetings → direct response
        │       ├─ identity/model questions → actual installed model info
        │       ├─ known facts → apple colors, physics constants, etc.
        │       ├─ math expressions → evaluated answer
        │       └─ reasoning/coding/unknown → buildModelNeededResponse()
        │           "⚠️ No Language Model Installed — here's how to download one..."
        │
        └─ messageDao streaming update (same as RAG path)
```

---

## 6. Supported AI Models

### Embedding Models
| Model | Format | Size | Dimension | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2.Q4_K_M.gguf` | GGUF | ~45 MB | 384 | Auto-downloaded from HuggingFace |
| Any GGUF embedding model | GGUF | varies | varies | Place in `filesDir/models/embeddings/` |
| *(none)* | — | 0 MB | 768 | Improved hash fallback (automatic) |

### Language Models (User-Installed)
| Model | Format | Size | Performance |
|---|---|---|---|
| Qwen 2.5 0.5B Q4 | GGUF | ~400 MB | Fast, good for mobile |
| Qwen 2.5 1.5B Q4 | GGUF | ~1 GB | Better quality |
| Llama 3.2 1B Q4 | GGUF | ~700 MB | Good general model |
| DeepSeek R1 1.5B | GGUF | ~1 GB | Strong reasoning |
| Any GGUF LLM | GGUF | varies | Place in `filesDir/models/llm/` |

Model files are placed in Android internal storage: `context.filesDir/models/llm/`. The active model is persisted in `SharedPreferences("docai_models")["selected_llm_model"]`.

---

## 7. Memory & Performance Architecture

### Memory Budget
```
M_total = M_model + M_index + M_KV + M_runtime ≤ 2.0 GB

M_model:   LLM GGUF loaded via mmap (minimal RAM copy)
M_index:   All chunk embeddings in Room (on disk, loaded on demand)
M_KV:      nCtx=4096 × nLayers × KV_head_dim × dtype_size
M_runtime: JVM heap + Android system overhead
```

### Key Latency Optimizations

| Technique | Where Applied | Effect |
|---|---|---|
| Incremental ingestion cache | DocumentRepository | Skip re-indexing already-processed docs |
| GlobalSummary short-circuit | ChatRepository | 0ms vector search for summary queries |
| Lexical pre-filter (Stage 1) | VectorStore | Reduces semantic search from N→40 candidates |
| Selective Content Reduction | SelectiveContentReducer | 25-40% fewer prompt tokens → lower TTFT |
| Context budget cap (2400 chars) | SCR.formatCompressedContext | Prevents prompt overflow in small SLMs |
| Memory-governed TopK | MemoryGovernor + VectorStore | Prevents OOM on low-RAM devices |
| mmap model loading | LlamaContext (`useMmap=true`) | No full RAM copy of model weights |

---

## 8. Package Structure

```
com.siddu.offlinedocai/
│
├── MainActivity.kt           — Entry point, Compose NavHost
├── MainViewModel.kt          — Central VM, DB init, model management, StateFlows
│
├── ai/
│   ├── chunking/
│   │   └── TextChunker.kt          — Sentence-aware sliding window chunker
│   ├── compression/
│   │   └── SelectiveContentReducer.kt  — SCR sentence scoring & context formatting
│   ├── embeddings/
│   │   ├── EmbeddingModel.kt       — Interface: getEmbedding(text): FloatArray
│   │   └── GGUFEmbeddingEngine.kt  — LlamaContext(embedding=true) + 768-dim fallback
│   ├── export/
│   │   └── ChatExporter.kt         — JSON chat export (conversation → file)
│   ├── governor/
│   │   └── MemoryGovernor.kt       — ρ-based adaptive config (topK, SCR ratio)
│   ├── llm/
│   │   ├── LocalLLM.kt             — Interface: generate(prompt, systemPrompt?)
│   │   ├── GGUFLLM.kt              — LlamaContext inference + intelligent fallback
│   │   ├── OllamaLLM.kt            — Dev-only Ollama HTTP bridge (disabled)
│   │   └── AvailableLlmModels.kt   — Known model registry (name, URL, size)
│   ├── profiler/
│   │   └── DocumentProfiler.kt     — RAKE keywords, section detection, domain inference
│   ├── queue/
│   │   ├── QuestionQueueManager.kt — Batch/sequential question execution
│   │   ├── QuestionQueueParser.kt  — Parse numbered/bulleted question lists
│   │   └── QuestionQueueState.kt   — State sealed class for queue UI
│   ├── reformulation/
│   │   └── QueryReformulator.kt    — Intent classification + search string rewriting
│   └── retrieval/
│       └── VectorStore.kt          — 2-stage hybrid retrieval (Lexical → Semantic)
│
├── data/
│   ├── AppDatabase.kt              — Room @Database (version 3)
│   ├── Converters.kt               — Room TypeConverters
│   ├── FloatArrayConverter.kt      — FloatArray ↔ ByteArray via ByteBuffer
│   ├── dao/
│   │   ├── ChunkDao.kt             — insertAll, getChunksForDocument, etc.
│   │   ├── ConversationDao.kt      — CRUD for conversations
│   │   ├── DocumentDao.kt          — CRUD + getByUri (incremental cache)
│   │   └── MessageDao.kt           — insert/update/flow for messages
│   └── entities/
│       ├── ChunkEntity.kt          — id, documentId, pageNumber, text, embedding (ByteArray)
│       ├── ConversationEntity.kt   — id, title, createdAt, documentId
│       ├── DocumentEntity.kt       — id, name, uri, type, pageCount, status, profileJson
│       └── MessageEntity.kt        — id, conversationId, role, content, sources, thinkingDurationMs
│
├── domain/
│   ├── rag/
│   │   └── DocumentProcessor.kt   — PDF/DOCX/TXT extraction, boilerplate stripping
│   └── repository/
│       ├── ChatRepository.kt       — RAG orchestrator, streaming, state machine
│       ├── DocumentRepository.kt   — Ingestion pipeline coordinator
│       └── ModelDownloader.kt      — Background HTTP model download with progress
│
└── (JNI Java source)
    └── org.nehuatl.llamacpp/
        └── LlamaContext.java       — JNI bridge to librnllama.so
```

---

## 9. Key Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| **llama.cpp JNI over MediaPipe/ML Kit** | Full GGUF model support, quantization (Q4/Q8), full control over generation params | Larger `.so` bundle, more complex integration |
| **Room SQLite for vector store** | Zero external dependency, offline-first, embedding stored as BLOB | No approximate nearest neighbor (ANN) index; full table scan for all chunks. Acceptable for <10k chunks |
| **Hash-based fallback embedding (768-dim)** | Ensures retrieval works even without a downloaded model | Inferior semantic discrimination vs real neural embeddings; cosine scores are less meaningful |
| **Sentence-aware chunking (not fixed-token)** | Preserves semantic completeness; no mid-sentence breaks | Slightly uneven chunk sizes |
| **Cross-page chunk spanning** | Preserves context that crosses physical page breaks | Page attribution becomes approximate (uses central sentence's page) |
| **Deterministic intent classification** | Zero latency, no neural classifier needed | Less nuanced than ML intent detection; may misclassify edge cases |
| **GlobalSummary short-circuit** | Eliminates vector search for overview queries using cached profile | Profile may not reflect full document depth for complex multi-topic docs |
| **GGUF-only model format** | Standardized, quantized, compatible with llama.cpp | No support for PyTorch/safetensors/ONNX LLMs (though ONNX Runtime is a dependency) |
| **Simulated token streaming** | Live-typing UX without true streaming callback integration | Adds `words.size × 30ms` latency to response display |

---

## 10. Known Limitations

| Limitation | Impact | Potential Fix |
|---|---|---|
| No ANN index (full table scan) | Slow for >10k chunks / very large document sets | Add FAISS, ScaNN, or Room FTS5-based index |
| Embedding dimension mismatch on re-import | Old 384-dim chunks get cosine=0; lexical-only scoring | Re-import existing documents to generate correct-dim embeddings |
| Simulated streaming (not true token-by-token) | Slightly inflated apparent latency | Integrate `onPartialCompletion` callback into coroutine flow |
| DOCX has no page boundaries | Entire DOCX is treated as page 1 | Add paragraph-count-based virtual pagination |
| Document Profile built once at ingestion | Profile doesn't update if document is re-indexed | Invalidate and rebuild profile on re-import |
| No multi-document RAG | Each conversation links to one document | Support multi-document context aggregation |
| Question Queue is sequential | Batch questions run one-by-one | Parallelize independent questions where context allows |
| No persistent LlamaContext across app restarts | Model reloads on every cold start | Implement a foreground service or model caching layer |
