import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from config import SOURCE_TEXT_FILE, GEMINI_MODELS_PRIORITY, retry_on_quota

# 1. Setup Gemini with Fallback
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

WORKING_MODEL = None

def get_working_model():
    global WORKING_MODEL
    if WORKING_MODEL:
        return WORKING_MODEL
        
    for model_name in GEMINI_MODELS_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            model.generate_content("test", generation_config={"max_output_tokens": 1})
            WORKING_MODEL = model
            print(f"Using model for Baseline RAG: {model_name}")
            return WORKING_MODEL
        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}. Trying next...")
            continue
    raise RuntimeError("All models hit quota.")

# 2. Setup Vector Search
model_st = SentenceTransformer('all-MiniLM-L6-v2')

# Load and chunk text (simple sentence chunking)
with open(SOURCE_TEXT_FILE, "r") as f:
    text = f.read()
chunks = [s.strip() for s in text.split('.') if s.strip()]

# Create FAISS index
embeddings = model_st.encode(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

def retrieve(query, k=2):
    query_vec = model_st.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

@retry_on_quota(max_retries=5, initial_wait=10)
def baseline_rag_query(query):
    retrieved_chunks = retrieve(query)
    context = "\n".join(retrieved_chunks)
    
    prompt = f"Answer the following question using the provided context only.\nContext:\n{context}\n\nQuestion: {query}"
    
    llm = get_working_model()
    response = llm.generate_content(prompt)
    
    return {
        "answer": response.text.strip(),
        "contexts": retrieved_chunks
    }

if __name__ == "__main__":
    # Test run
    test_q = "Who founded Quantum Dynamics?"
    result = baseline_rag_query(test_q)
    print(f"Question: {test_q}\nAnswer: {result['answer']}\nContexts: {result['contexts']}")
