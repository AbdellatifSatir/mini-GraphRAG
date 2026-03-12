# GraphRAG Hands-on Project State

## 🎯 Current Objective
Building a modular, automated GraphRAG (Graph Retrieval-Augmented Generation) pipeline using Python, NetworkX, and Gemini 2.5 Flash.

## 🛠️ Tech Stack
- **LLM:** Google Gemini 2.5 Flash (via `google-generativeai`)
- **Graph Engine:** NetworkX (Python)
- **Persistence:** GraphML format (`knowledge_graph.graphml`)
- **Community summaries:** `community_summaries.json`
- **Environment:** Python 3.10+, `python-dotenv`

## 🏗️ Project Architecture
1.  **`source_text.txt`**: The raw knowledge base (unstructured text).
2.  **`gemini_kg_builder.py`**: 
    - Checks for existing graph.
    - Extracts Entities/Relations in JSON.
    - Groups nodes into communities using Louvain and generates summaries.
    - Builds and saves the GraphML file.
3.  **`graph_rag_assistant.py` (The Unified Brain)**:
    - **Step 1 (Query Classification):** Determines if a query is LOCAL (facts) or GLOBAL (themes).
    - **Step 2 (Semantic Resolution):** Maps vague query terms to exact Graph Nodes using LLM.
    - **Step 3 (Multi-Hop Retrieval):** Performs a 2-hop traversal (`nx.ego_graph`) for local context.
    - **Step 4 (Global Context):** Uses community summaries for broad thematic answers.
    - **Step 5 (Grounded Generation):** Answers the query using *only* the retrieved context.

## ✅ Accomplishments
- [x] Automated KG Construction from text.
- [x] Persistent storage (Phase 2).
- [x] Unified RAG Assistant (Phase 3).
- [x] Multi-Hop Retrieval (2-hop depth).
- [x] Semantic Entity Resolution (mapping vague terms to nodes).
- [x] **Community Detection (Phase 4):** Integrated Louvain clustering and thematic summarization for global queries.
- [x] **Validated System:** Confirmed both local and global retrieval pathways are functional with Gemini 2.5 Flash.

## 🚀 Next Steps (Phase 5 & Beyond)
- [ ] **Vector Hybrid Search:** Combine Graph retrieval with Vector embeddings (FAISS/ChromaDB) for "Fuzzy" semantic search and improved entity resolution.
- [ ] **Neo4j Integration:** Scale the project from a file-based graph to a professional graph database.

## 💡 How to Resume
To continue this session, provide the AI with the contents of `source_text.txt`, `gemini_kg_builder.py`, and `graph_rag_assistant.py`. Ask to proceed with the **"Vector Hybrid Search"** phase as outlined in `GEMINI.md`.
