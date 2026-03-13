# GraphRAG Hands-on Project State

## 🎯 Current Objective
Building a modular, automated GraphRAG (Graph Retrieval-Augmented Generation) pipeline using Python, NetworkX, and Gemini 1.5 Flash.

## 🛠️ Tech Stack
- **LLM:** Google Gemini 1.5 Flash (`gemini-flash-latest`)
- **Graph Engine:** NetworkX (Python)
- **Vector DB:** FAISS (for hybrid search)
- **Embeddings:** `all-MiniLM-L6-v2`
- **Persistence:** Neo4j (Primary), GraphML format (`knowledge_graph.graphml`) (Backup)
- **Community summaries:** Hierarchical `community_summaries.json`
- **Environment:** Python 3.10+, `python-dotenv`

## 🏗️ Project Architecture
1.  **`source_text.txt`**: The raw knowledge base (unstructured text).
2.  **`gemini_kg_builder.py`**: 
    - Checks for existing graph.
    - Extracts Entities/Relations in JSON.
    - **Hierarchical Clustering:** Groups nodes into communities (Level 0) and then creates a "Global Overview" (Level 1).
    - **Neo4j Sync:** Mirrors the NetworkX graph into a Neo4j instance for persistent, queryable storage.
3.  **`vector_indexer.py`**:
    - Generates embeddings for all graph nodes and stores them in a FAISS index for fuzzy searching.
4.  **`graph_rag_assistant.py` (The Unified Brain)**:
    - **Step 1 (Query Classification):** Determines if a query is LOCAL (facts) or GLOBAL (themes).
    - **Step 2 (Hybrid Resolution):** Uses Vector Search (FAISS) + LLM refinement to map query terms to exact Graph Nodes.
    - **Step 3 (Neo4j Multi-Hop Retrieval):** Performs a 2-hop traversal directly in Neo4j using Cypher.
    - **Step 4 (Intelligent Global Context):** Uses Gemini to decide if a query needs a high-level (Level 1) or detailed (Level 0) summary.
    - **Step 5 (Grounded Generation):** Answers the query using *only* the retrieved context.

## ✅ Accomplishments
- [x] Automated KG Construction from text.
- [x] Persistent storage (Phase 2).
- [x] Unified RAG Assistant (Phase 3).
- [x] Multi-Hop Retrieval (2-hop depth).
- [x] Semantic Entity Resolution (mapping vague terms to nodes).
- [x] **Phase 4: Community Detection:** Integrated Louvain clustering and thematic summarization.
- [x] **Phase 5: Vector Hybrid Search:** Added FAISS for "fuzzy" node mapping and semantic search.
- [x] **Phase 6: Hierarchical Summarization:** Implemented multi-level summaries (Level 0/Level 1) and intelligent context routing.
- [x] **Phase 7: Neo4j Integration:** Migrated from file-based GraphML to a professional graph database for retrieval.
- [x] **Phase 7.5: Community Migration:** Stored hierarchical summaries directly as `:Community` nodes in Neo4j. Refactored Assistant to retrieve global context via Cypher.

## 🚀 Next Steps (Phase 8 & Beyond)
- [ ] **Phase 8: Agentic Graph Traversal:** Give the LLM tools to "walk" the graph dynamically by generating its own Cypher queries.
- [ ] **Phase 9: Evaluation (RAGAS):** Measure and compare GraphRAG performance against standard RAG.

## 💡 How to Resume
To continue this session, provide the AI with the contents of `source_text.txt`, `gemini_kg_builder.py`, and `graph_rag_assistant.py`. Ask to proceed with the **"Neo4j Integration"** phase as outlined in `GEMINI.md`.
