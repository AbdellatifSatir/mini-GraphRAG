import os
import json
import google.generativeai as genai
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Configuration & Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# Load Embedding Model for Vector Search
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_type(query):
    """Determines if the query is 'Local' (specific entities) or 'Global' (broad themes)."""
    prompt = f"""
    Analyze this user question and classify it as either 'LOCAL' or 'GLOBAL'.
    - LOCAL: Questions about specific people, organizations, or facts (e.g., "Who founded X?").
    - GLOBAL: Questions about high-level summaries, themes, or "what is this document about?".
    
    Respond ONLY with the word 'LOCAL' or 'GLOBAL'.
    
    Question: {query}
    Classification:"""
    
    response = model.generate_content(prompt)
    classification = response.text.strip().upper()
    
    # Extract only the key word to be safe
    if "GLOBAL" in classification and "LOCAL" not in classification:
        return "GLOBAL"
    elif "LOCAL" in classification and "GLOBAL" not in classification:
        return "LOCAL"
    # Fallback: Check if GLOBAL is the last word or alone
    return "GLOBAL" if "GLOBAL" in classification else "LOCAL"

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

def get_local_context(entities, hops=2):
    """Retrieves context using Neo4j Cypher queries for multi-hop traversal."""
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        return "Neo4j connection not configured."

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # 1. Fetch all entity IDs from Neo4j to help with mapping
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.id AS id")
        all_nodes = [record["id"] for record in result]
    
    # 2. Map query entities to graph nodes
    resolved_nodes = map_entities_to_nodes(entities, all_nodes)
    print(f"   Resolved Nodes: {resolved_nodes}")
    
    if not resolved_nodes:
        driver.close()
        return ""

    context_triples = []
    
    # 3. Perform Multi-Hop Traversal in Neo4j
    with driver.session() as session:
        for target_node in resolved_nodes:
            # Find all nodes and relationships within 'hops' distance
            # We use RELATED_TO and extract the 'type' property for the relation label
            query = """
            MATCH (n:Entity {id: $node_id})
            MATCH (n)-[r:RELATED_TO*1..%d]-(m:Entity)
            UNWIND r AS rel
            RETURN startNode(rel).id AS source, endNode(rel).id AS target, rel.type AS relation
            """ % hops
            
            result = session.run(query, node_id=target_node)
            for record in result:
                context_triples.append(f"{record['source']} --[{record['relation']}]--> {record['target']}")
    
    driver.close()
    
    final_context = "\n".join(list(set(context_triples)))
    if final_context:
        print(f"   Context Triples Found:\n{final_context}")
    else:
        print("   No context triples found.")
        
    return final_context

def get_global_context(query, summary_file="community_summaries.json"):
    """Retrieves context from hierarchical community summaries."""
    if not os.path.exists(summary_file): return "No global summaries available."
    
    with open(summary_file, "r", encoding="utf-8") as f:
        hierarchical_data = json.load(f)
    
    # Use Gemini to decide the required level of detail
    level_0_count = len(hierarchical_data.get("level_0", {}))
    level_1_summary = hierarchical_data.get("level_1", {}).get("0", {}).get("summary", "")
    
    prompt = f"""
    A user has asked a global question. Determine if the answer requires a 'HIGH-LEVEL' overview or 'DETAILED' themes.
    
    Question: {query}
    High-level Overview: {level_1_summary}
    
    Respond with either 'HIGH-LEVEL' or 'DETAILED'.
    Decision:"""
    
    response = model.generate_content(prompt)
    decision = response.text.strip().upper()
    print(f"   Global Detail Decision: {decision}")

    if "HIGH-LEVEL" in decision:
        return f"Global Overview: {level_1_summary}"
    else:
        context = []
        for comm_id, data in hierarchical_data["level_0"].items():
            context.append(f"Detailed Theme {comm_id}: {data['summary']}")
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
    print("--- Welcome to the Advanced GraphRAG Assistant (Phase 7 - Neo4j) ---")
    
    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() == 'exit': break

        # 1. Determine Query Type
        q_type = get_query_type(query)
        print(f"   Query Type: {q_type}")

        if "GLOBAL" in q_type:
            print("   Retrieving Global Context...")
            context = get_global_context(query)
        else:
            print("   Recognizing entities...")
            entities = extract_entities_from_query(query)
            print(f"   Retrieving Local Context for: {entities} from Neo4j...")
            context = get_local_context(entities)
        
        # 2. Generate Answer
        print("   Generating grounded answer...")
        answer = generate_grounded_answer(query, context, q_type)
        print(f"\nFINAL ANSWER:\n{answer}\n" + "-"*30)

if __name__ == "__main__":
    main()
