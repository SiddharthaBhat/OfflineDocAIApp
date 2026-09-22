# Research Versioning Log: State 002
**Timestamp**: August 2026  
**System Milestone**: Version 2.0 (Enhanced RAG Retrieval & Multi-SLM Model Management)  
**Target Environment**: Android (On-Device Inference, Dynamic GGUF Selection & Storage Management)

---

## 1. Research Objectives for State 002
State 002 directly resolves the limitations identified in [state_001.md](state_001.md):
1. **Fixing Suboptimal Retrieval ("Messed Up Retrieval")**: Replacing naive fallback retrieval with TF-IDF / BM25 term weighting and document-wide uniform sampling for generic queries.
2. **Handling Generic & High-Level Document Queries**: Providing structured, non-spam summary synthesis for abstract user questions (e.g., *"What is this doc about?"*, *"Summarize this file"*).
3. **App-Level SLM Model Lifecycle Management**: Enabling on-device download, active model switching, and deletion of downloaded `.gguf` weights directly from the user interface.

---

## 2. Technical Enhancements & Architectural Changes

### A. Dynamic Query Routing & Global Sampling ([VectorStore.kt](../app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/VectorStore.kt))
- **Generic Query Detection**: Query strings containing tokens like `about`, `summary`, `summarize`, `overview`, `main`, `tell`, or queries $\le 3$ words are flagged as high-level intent.
- **Uniform Stratified Sampling**: Rather than pulling localized top-$K$ chunks, the retriever computes a step interval $S = \frac{N_{\text{total}}}{K}$ to sample representative chunks evenly across the entire document length.

### B. Inverse Document Frequency (TF-IDF) Lexical Ranking ([VectorStore.kt](../app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/VectorStore.kt))
- For targeted queries where dense vector similarity returns zero/mock values, lexical retrieval computes exact Term Frequency (TF) combined with Inverse Document Frequency (IDF):
$$\text{IDF}(t) = \ln\left(\frac{N + 1.0}{\text{DF}(t) + 0.5}\right) + 1.0$$
- Filters out standard English stop words (`what`, `when`, `where`, `how`, `the`, `is`, `and`, etc.) to prevent high-frequency noise from overwhelming scoring.

### C. Clean Synthesis Generator ([GGUFLLM.kt](../app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt))
- Detects global overview prompts and formats the context into bulleted key findings with page attribution tips instead of dumping raw chunk text.

### D. On-Device SLM Model Management ([MainViewModel.kt](../app/src/main/kotlin/com/siddu/offlinedocai/MainViewModel.kt) & [MainActivity.kt](../app/src/main/kotlin/com/siddu/offlinedocai/MainActivity.kt))
- **File System Inspection**: Scans `filesDir/models/llm/` for GGUF model files.
- **Model Switcher**: Allows users to select between multiple downloaded SLM weights (e.g., Qwen 2.5 0.5B vs. Llama 3.2 1B).
- **Storage Cleanup**: Direct option in the Settings UI to delete downloaded `.gguf` weights and free up disk space.

---

## 3. Comparison Summary: State 001 vs. State 002

| Metric / Feature | State 001 Baseline | State 002 Implementation |
| :--- | :--- | :--- |
| **Generic Query Support** (*"What is this doc about?"*) | ❌ Fails (returns random/irrelevant local chunk) | ✅ **Supported** (Uniform sampling & synthesized summary) |
| **Retrieval Accuracy** | Low (simple word match fallback) | **Enhanced** (TF-IDF + Stop-word filtering) |
| **Model Management** | Single static file check | ✅ **Multi-SLM support: Download, Switch & Delete UI** |
| **Response Cleanliness** | Noise/cluttered text output | Structured Markdown output with clean bullet points |

---

## 4. Next Research Directions (State 003 Considerations)
- Native JNI integration for llama.cpp / GGUF execution directly on mobile GPU/NPU.
- Advanced BM25 + Reciprocal Rank Fusion (RRF) dense hybrid ranking.
