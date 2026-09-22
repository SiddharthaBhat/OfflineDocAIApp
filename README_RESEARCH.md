# Offline Doc AI App - Research & Versioning History

This directory tracks the architectural iterations, experimental setups, RAG pipeline changes, and performance benchmarking for research publications.

## Index of Versioned States

- **[State 001 - Initial Offline RAG Architecture & Baseline Mechanics](docs/state_001.md)**: Details the baseline document parsing, sliding-window chunking, vector storage with Room DB, cosine similarity calculation, and fallback retrieval strategies.
- **[State 002 - Enhanced Retrieval & SLM Lifecycle Management](docs/state_002.md)**: Implements dynamic query routing, uniform stratified sampling for generic queries (*"What is this doc about?"*), TF-IDF lexical search, clean summary synthesis, and on-device model management (switching, downloading, and deleting SLMs).

