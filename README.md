# 🧠 Mini-GraphRAG: Modular Knowledge Graph RAG Pipeline

A modular, automated GraphRAG (Graph Retrieval-Augmented Generation) pipeline built with **Python**, **NetworkX**, **Neo4j**, and **Google Gemini 1.5/2.0 Flash**. This project implements a hybrid retrieval system that combines the factual precision of a Knowledge Graph with the thematic understanding of Hierarchical Community Detection.

---

## 🚀 Project Overview
Traditional RAG often struggles with global "big picture" questions (e.g., "What are the main themes?") and multi-hop reasoning. This project solves those issues by:
1.  **Extracting** a Knowledge Graph (Entities & Relations) from unstructured text using LLMs.
2.  **Clustering** nodes into communities using the Louvain algorithm.
3.  **Summarizing** those communities into a hierarchical structure (Level 0 and Level 1).
4.  **Indexing** nodes using FAISS for semantic "fuzzy" entity resolution.
5.  **Retrieving** context via a hybrid path: Neo4j (Multi-hop) + Vector Search + Hierarchical Summaries.

---

## 🛠️ Tech Stack
- **LLM:** Google Gemini 1.5 Flash / 2.0 Flash (via `google-generativeai`)
- **Graph Engine:** NetworkX (Logic) & Neo4j (Persistence/Retrieval)
- **Vector DB:** FAISS (for node-level semantic search)
- **Embeddings:** `all-MiniLM-L6-v2` (via `sentence-transformers`)
- **Environment:** Python 3.10+, `python-dotenv` for configuration.

---

## 🏗️ System Architecture

### 1. Knowledge Graph Builder (`gemini_kg_builder.py`)
- Processes `source_text.txt`.
- Uses Gemini to extract a JSON-formatted KG.
- Implements **Louvain Community Detection** to group entities.
- Generates a two-layer summary hierarchy (Base Communities & Global Overview).
- Syncs the entire graph to a **Neo4j** instance.

### 2. Vector Indexer (`vector_indexer.py`)
- Creates a FAISS index of all node names.
- Maps user query terms (e.g., "Sarah") to exact graph nodes (e.g., "Sarah Chen").

### 4. Agentic GraphRAG Assistant (`agentic_graph_rag.py` & `agent_tools.py`)
- **Reasoning Loop:** Implements a Thought-Action-Observation loop (ReAct) for dynamic exploration.
- **Dynamic Cypher Generation:** The LLM generates and executes Cypher queries based on the graph schema.
- **Tools:**
  - `get_schema()`: Inspects database structure.
  - `resolve_entities()`: Maps vague query terms to graph nodes.
  - `run_cypher()`: Executes multi-hop traversals.

---

## ✅ Accomplishments (Phases 1-8)
- [x] **Phase 1-2:** Automated KG construction and GraphML persistence.
- [x] **Phase 3-4:** Community detection and thematic summarization.
- [x] **Phase 5:** FAISS-based semantic entity resolution.
- [x] **Phase 6:** Hierarchical context routing (Local vs. Global).
- [x] **Phase 7:** Full Neo4j integration for persistent retrieval.
- [x] **Phase 7.5:** Community Migration (Summaries in Neo4j).
- [x] **Phase 8:** Agentic Graph Traversal (Dynamic Cypher generation and reasoning loop).

---

## 🚀 Next Steps
- [ ] **Phase 9: Evaluation:** benchmark performance using the RAGAS framework.

---

## 🛠️ Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Environment:**
    Create a `.env` file with your `GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.
3.  **Build the Graph:**
    ```bash
    python gemini_kg_builder.py
    ```
4.  **Index the Nodes:**
    ```bash
    python vector_indexer.py
    ```
5.  **Run the Assistant:**
    ```bash
    python graph_rag_assistant.py
    ```

---
*Created by Abdellatif Satir - 2026*
