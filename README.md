# Mini-GraphRAG: Hybrid Knowledge Graph Assistant

A modular, automated GraphRAG (Graph Retrieval-Augmented Generation) pipeline that combines structured knowledge graphs with semantic vector search and LLM-driven reasoning.

## 🚀 The Idea
Standard RAG often struggles with complex relationships and high-level thematic queries. This project implements a **GraphRAG** approach using:
- **Knowledge Graphs (NetworkX):** To capture explicit relationships between entities (e.g., *Sarah Chen* founded *Quantum Dynamics*).
- **Vector Search (FAISS):** To enable "fuzzy" semantic matching (e.g., finding "Quantum Dynamics" when the user asks about "the Berlin tech firm").
- **LLM Reasoning (Gemini 2.5 Flash):** To classify queries, extract context, and generate grounded, fact-based answers.

## ✅ Current Progress

### Phase 1-3: Automated KG & Local RAG
- **Automated Construction:** Transforms raw text into a structured Knowledge Graph using Gemini.
- **Local Retrieval:** Uses 2-hop graph traversal to gather deep context for specific entity-based questions.
- **Grounded Generation:** Ensures answers are strictly derived from the graph to prevent hallucinations.

### Phase 4: Community Detection (Global RAG)
- **Louvain Clustering:** Automatically groups related nodes into thematic communities.
- **Global Summaries:** Generates high-level summaries for each community, allowing the assistant to answer "What is this document about?" type questions.

### Phase 5: Vector Hybrid Search
- **FAISS Integration:** Node names are embedded and stored in a vector index.
- **Semantic Resolution:** Combines vector similarity with LLM refinement to map vague query terms to exact graph nodes.

## 🛠️ Tech Stack
- **LLM:** Google Gemini 2.5 Flash
- **Graph Engine:** NetworkX
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Clustering:** Louvain Community Detection

## ⚙️ Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Build the Graph:**
   Place your text in `source_text.txt` and run:
   ```bash
   python gemini_kg_builder.py
   ```

4. **Index for Hybrid Search:**
   Generate the vector index for node names:
   ```bash
   python vector_indexer.py
   ```

5. **Start the Assistant:**
   ```bash
   python graph_rag_assistant.py
   ```

## 🚀 Roadmap (Next Steps)
- [ ] **Phase 6: Hierarchical Summarization:** Implement "summaries of summaries" to handle massive datasets and improve global reasoning.
- [ ] **Phase 7: Neo4j Integration:** Scale the project from a file-based system to a professional graph database.
- [ ] **Phase 8: Agentic Graph Traversal:** Give the LLM tools to "walk" the graph dynamically based on query requirements.
