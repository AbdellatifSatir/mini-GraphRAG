import os
import json
import networkx as nx
import google.generativeai as genai
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. Configuration & Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Load Embedding Model for Vector Search
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_type(query):
    """Determines if the query is 'Local' (specific entities) or 'Global' (broad themes)."""
    prompt = f"""
    Analyze this user question and classify it as either 'LOCAL' or 'GLOBAL'.
    - LOCAL: Questions about specific people, organizations, or facts (e.g., "Who founded X?").
    - GLOBAL: Questions about high-level summaries, themes, or "what is this document about?".
    
    Question: {query}
    Classification (LOCAL/GLOBAL):"""
    
    response = model.generate_content(prompt)
    return response.text.strip().upper()

def extract_entities_from_query(query):
    """Asks Gemini to identify the main entities mentioned in the user's question."""
    prompt = f"""
    Given the following user question, identify the main entities (names, organizations, locations).
    Respond ONLY with a comma-separated list of entities.
    
    Question: {query}
    Entities:"""
    
    response = model.generate_content(prompt)
    entities = [e.strip() for e in response.text.split(",")]
    return entities

def vector_search_nodes(query_entities, index_file="node_index.faiss", mapping_file="node_mapping.pkl", top_k=2):
    """Finds the most similar nodes in the graph using FAISS."""
    if not os.path.exists(index_file) or not os.path.exists(mapping_file):
        return []

    index = faiss.read_index(index_file)
    with open(mapping_file, "rb") as f:
        all_nodes = pickle.load(f)

    resolved_nodes = []
    for entity in query_entities:
        query_embedding = embed_model.encode([entity]).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        
        for idx in indices[0]:
            if idx != -1:
                resolved_nodes.append(all_nodes[idx])
    
    return list(set(resolved_nodes))

def map_entities_to_nodes(query_entities, graph_nodes):
    """
    Hybrid Entity Resolution:
    1. Use Vector Search (FAISS) to find top candidates.
    2. Use Gemini to refine and confirm the mapping.
    """
    if not query_entities or not graph_nodes:
        return []

    # Step 1: Vector Search for candidates
    candidates = vector_search_nodes(query_entities)
    
    if not candidates:
        return []

    # Step 2: LLM refinement
    prompt = f"""
    You are an Entity Resolver. Map the user's 'Query Entities' to the most likely 'Graph Nodes' from the candidate list below.
    Only return the exact names from the 'Graph Nodes' list, separated by commas.
    If no match is found for an entity, ignore it.

    Query Entities: {", ".join(query_entities)}
    Graph Nodes (Candidates): {", ".join(candidates)}

    Mapped Nodes:"""
    
    response = model.generate_content(prompt)
    resolved_nodes = [n.strip() for n in response.text.split(",") if n.strip() in graph_nodes]
    return resolved_nodes

def get_local_context(entities, graph_file="knowledge_graph.graphml", hops=2):
    """Retrieves context using 2-hop graph traversal."""
    if not os.path.exists(graph_file): return ""
    G = nx.read_graphml(graph_file)
    all_nodes = list(G.nodes())
    
    # RESOLUTION STEP: Use Hybrid Mapping
    resolved_nodes = map_entities_to_nodes(entities, all_nodes)
    print(f"   Resolved Nodes: {resolved_nodes}")
    
    context_triples = []
    for target_node in resolved_nodes:
        ego = nx.ego_graph(G, target_node, radius=hops, undirected=True)
        for u, v, d in ego.edges(data=True):
            relation = d.get('relation', 'connected to')
            context_triples.append(f"{u} --[{relation}]--> {v}")
    
    return "\n".join(list(set(context_triples)))

def get_global_context(summary_file="community_summaries.json"):
    """Retrieves context using community summaries."""
    if not os.path.exists(summary_file): return "No global summaries available."
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)
    
    context = []
    for comm_id, data in summaries.items():
        context.append(f"Theme {comm_id}: {data['summary']}")
    
    return "\n".join(context)

def generate_grounded_answer(query, context, q_type):
    """Uses Gemini to answer the query using the provided context."""
    if not context:
        return "I'm sorry, I couldn't find any relevant information in the knowledge graph."

    role = "Local Fact Assistant" if q_type == "LOCAL" else "Global Theme Assistant"
    
    prompt = f"""
    You are a {role}. Answer the user's question ONLY using the provided context.
    
    Context:
    {context}

    User Question: {query}
    
    Answer:"""
    
    response = model.generate_content(prompt)
    return response.text

def main():
    print("--- Welcome to the Advanced GraphRAG Assistant (Phase 5 - Hybrid) ---")
    
    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() == 'exit': break

        # 1. Determine Query Type
        q_type = get_query_type(query)
        print(f"   Query Type: {q_type}")

        if "GLOBAL" in q_type:
            print("   Retrieving Global Context...")
            context = get_global_context()
        else:
            print("   Recognizing entities...")
            entities = extract_entities_from_query(query)
            print(f"   Retrieving Local Context for: {entities}...")
            context = get_local_context(entities)
        
        # 2. Generate Answer
        print("   Generating grounded answer...")
        answer = generate_grounded_answer(query, context, q_type)
        print(f"\nFINAL ANSWER:\n{answer}\n" + "-"*30)

if __name__ == "__main__":
    main()
