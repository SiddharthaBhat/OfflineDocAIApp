# Research Versioning Log: State 001
**Timestamp**: August 2026  
**System Baseline**: Version 1.0 (Initial On-Device RAG Implementation)  
**Target Environment**: Android (On-Device Inference & Vector Search)

---

## 1. Overview & Research Context
This document serves as the foundational snapshot (**State 001**) for the **Offline Doc AI** project. The goal of this research project is to design, implement, and benchmark a completely privacy-preserving, on-device Retrieval-Augmented Generation (RAG) system operating under standard Android hardware constraints without external server dependencies.

---

## 2. Document Processing & Ingestion Pipeline

### Supported Formats & Extraction Libraries
The current system ingests multi-format local user files directly on-device:
- **PDF (.pdf)**: Extracted via `PDFBox-Android` (`com.tom_roush.pdfbox.pdmodel.PDDocument`). Pages are processed iteratively (`stripper.startPage` / `stripper.endPage`) to preserve physical page numbers for citation attribution.
- **DOCX (.docx)**: Extracted via `Apache POI` (`XWPFDocument`). Analyzed paragraph-by-paragraph. (Treated logically as Page 1 due to lack of standard page layout boundaries in DOCX structure).
- **Plain Text (.txt)**: Extracted line-by-line using standard Java/Kotlin IO (`BufferedReader`).

---

## 3. Text Chunking Strategy

### Algorithm Parameters
- **Chunker Type**: Sliding Word-Window (`TextChunker`)
- **Default Chunk Size ($L_c$)**: 400 words
- **Default Overlap ($O_c$)**: 80 words (20% overlap)
- **Tokenization Basis**: Regular expression whitespace splitting (`\\s+`)

### Rationale & Boundary Handling
The 400-word window size balances context preservation (sufficient for complete context in technical papers or legal text) while fitting comfortably within typical small-footprint LLM context windows (e.g., 2048 to 4096 tokens) alongside system prompts. The 80-word overlap mitigates context fragmentation across chunk boundaries.

---

## 4. Embedding Generation & Vector Store

### Embedding Architecture
- **Interface**: `EmbeddingModel`
- **Engine Implementation**: `GGUFEmbeddingEngine`
- **Model Path**: `filesDir/models/embeddings/*.gguf`
- **Dimensionality**: 384-dimensional vector outputs ($d = 384$)
- **Fallback / Prototype Behavior**: Returns deterministic zero-vectors when local `.gguf` weights are absent, transitioning gracefully to keyword fallback retrieval.

### Persistence Layer (`Room DB`)
Embeddings are flattened into `ByteArray` representations using byte buffers (`FloatArrayConverter`) and stored in SQLite via Room ORM:
- **`DocumentEntity`**: ID, file name, URI, timestamp, page count.
- **`ChunkEntity`**: Chunk ID, parent Document ID, page number, chunk index, raw text, and binary embedding blob (`ByteArray`).

---

## 5. Retrieval & Scoring Strategies

### 1. Vector Search (Cosine Similarity)
When valid floating-point embedding vectors are generated, retrieval ranks document chunks based on Cosine Similarity:
$$\text{Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### 2. Lexical / Keyword Matching (Hybrid Fallback)
To ensure robustness during local cold-start testing or when embedding models are unavailable/unloaded:
- If all vector similarity scores evaluate to `0.0`, `VectorStore` dynamically falls back to an exact token-match algorithm.
- Filters out standard English stop words (`what`, `when`, `where`, `how`, `this`, `that`, `there`, `their`, `them`, `with`, `from`).
- Ranks chunks based on term-frequency score across query keywords and selects the `Top-K` matching segments.

---

## 6. Local LLM Generation & Context Assembly

- **Local Inference Engine**: `GGUFLLM` & `OllamaLLM`
- **Prompt Structure**:
  ```text
  Context information is below:
  ---------------------
  [Page X]: <Text of Chunk 1>
  [Page Y]: <Text of Chunk 2>
  ---------------------
  Given the context information above, answer the question: <User Query>
  ```
- **Attribution**: Grounded in specific extracted page numbers (`ExtractedPage.pageNumber`), enabling direct citation references back to the source document pages.

---

## 7. Experimental Observations & Baseline Evaluation

### 🟢 Advantages & Positives (What Works Well)
1. **Low Latency & High Speed**: Local execution yields minimal end-to-end latency. Document chunking, SQLite storage, and local context generation execute rapidly without network bottlenecks.
2. **Deterministic Response Generation**: The pipeline reliably returns responses without crashing or hanging, ensuring standard baseline availability.
3. **Privacy & Offline Integrity**: Zero data egress; 100% on-device processing.

### 🔴 Critical Limitations & Failure Modes (Areas Needing Improvement)

1. **Suboptimal Retrieval Quality ("Messed Up Retrieval")**:
   - The top-$K$ retrieved chunks frequently contain irrelevant or low-quality text snippets ("clutter/noise"), leading to unhelpful or verbose responses stuffed with unnecessary details.
   - Fine-grained semantic matching is weak, resulting in irrelevant context being passed to the generator.

2. **Inability to Answer High-Level / Generic Queries**:
   - Queries like *"What is this document about?"*, *"Summarize this file"*, or *"What does it tell?"* fail under the current point-lookup RAG approach.
   - **Root Cause**: The current chunk retrieval algorithm looks for top-$K$ localized fragments ($K \approx 3\text{--}5$) rather than performing global document summarization or multi-chunk hierarchical aggregation.

3. **Noise Accumulation in Prompt Context**:
   - Irrelevant chunks included in the LLM prompt confuse the model, prompting it to output off-target explanations or verbatim low-value text.

---

## 8. Research Hypotheses for Version 002 Iterations

1. **Hierarchical / Summary-Aware Indexing**: Generate high-level document summaries during ingestion to answer global/abstract queries.
2. **Dense Vector Engine & Reranking Improvements**: Upgrade from simple fallback/un-normalized cosine similarity to hybrid BM25 + Vector search with Reciprocal Rank Fusion (RRF).
3. **Adaptive Chunking & Context Filtering**: Implement semantic boundaries (e.g., Markdown headers, paragraph breaks) and score thresholding to discard low-confidence chunks.

