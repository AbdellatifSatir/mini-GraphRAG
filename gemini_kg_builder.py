import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
from community import community_louvain
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
    raise ValueError("GEMINI_API_KEY not found in .env file. Please check .env.example.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

def sync_to_neo4j(G):
    """Uploads the NetworkX graph to a Neo4j instance."""
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print("⚠️ Neo4j credentials missing. Skipping sync.")
        return

    print(f"Syncing {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # 1. Sync Nodes
        for node, data in G.nodes(data=True):
            node_type = data.get("type", "Unknown")
            session.run(
                "MERGE (e:Entity {id: $id}) SET e.type = $type",
                id=node, type=node_type
            )
        
        # 2. Sync Edges
        for u, v, data in G.edges(data=True):
            relation = data.get("relation", "connected to")
            # Using Cypher's APOC or standard MERGE with dynamic relationship types is tricky.
            # We'll use a generic 'RELATED_TO' type and store the specific relation as a property.
            session.run(
                """
                MATCH (a:Entity {id: $source})
                MATCH (b:Entity {id: $target})
                MERGE (a)-[r:RELATED_TO {type: $relation}]->(b)
                """,
                source=u, target=v, relation=relation
            )
            
    driver.close()
    print("✅ Sync to Neo4j complete.")

def extract_kg_from_text(text):
    """Uses Gemini to extract entities and relationships in JSON format."""
    prompt = f"""
    Extract a Knowledge Graph from the following text. 
    Identify:
    1. Entities (Nodes): Provide a name and a type (e.g., Person, Org, Location).
    2. Relationships (Edges): Provide the source entity, target entity, and the relationship type.

    Respond ONLY with a JSON object in this format:
    {{
      "nodes": [
        {{"id": "Entity Name", "type": "Type"}}
      ],
      "edges": [
        {{"source": "Entity A", "target": "Entity B", "relation": "Relationship"}}
      ]
    }}

    Text:
    {text}
    """
    
    response = model.generate_content(prompt)
    
    # Simple JSON cleaning in case Gemini adds markdown backticks
    raw_text = response.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:-3].strip()

    return json.loads(raw_text)

def generate_hierarchical_summaries(G):
    """Groups nodes into communities, summarizes them, then groups summaries for a higher-level view."""
    # --- Level 0: Base Communities ---
    undirected_G = G.to_undirected()
    partition = community_louvain.best_partition(undirected_G)
    
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    
    level_0_summaries = {}
    print(f"Level 0: Detected {len(communities)} communities. Summarizing...")
    
    for comm_id, nodes in communities.items():
        edges = [f"{u} --[{d['relation']}]--> {v}" for u, v, d in G.edges(data=True) if u in nodes and v in nodes]
        
        prompt = f"Provide a one-sentence summary for this community:\nEntities: {', '.join(nodes)}\nRelationships: {', '.join(edges)}\nSummary:"
        response = model.generate_content(prompt)
        level_0_summaries[str(comm_id)] = {"nodes": nodes, "summary": response.text.strip()}

    # --- Level 1: Meta-Communities (Summaries of Summaries) ---
    # In a real-world scenario, you'd cluster the Level 0 communities based on their inter-connectivity.
    # For this small graph, we'll create a "Global Root" summary that combines Level 0.
    
    print("Level 1: Generating global overview...")
    combined_summaries = "\n".join([f"- {s['summary']}" for s in level_0_summaries.values()])
    
    prompt = f"Given these sub-summaries of a document, provide a high-level one-sentence overview of the entire document:\n{combined_summaries}\nGlobal Summary:"
    response = model.generate_content(prompt)
    level_1_summary = response.text.strip()

    hierarchical_data = {
        "level_0": level_0_summaries,
        "level_1": {"0": {"summary": level_1_summary, "children": list(level_0_summaries.keys())}}
    }
    
    with open("community_summaries.json", "w", encoding="utf-8") as f:
        json.dump(hierarchical_data, f, indent=4)
    print("Hierarchical summaries saved to 'community_summaries.json'.")

def build_kg_from_data(data):
    """Constructs a NetworkX graph from JSON data and saves it."""
    G = nx.DiGraph()

    # Add nodes
    for node in data.get("nodes", []):
        G.add_node(node["id"], type=node.get("type", "Unknown"))

    # Add edges
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "connected to"))

    # Save the graph
    nx.write_graphml(G, "knowledge_graph.graphml")
    print("Graph built and saved to 'knowledge_graph.graphml'.")
    
    # NEW: Generate Hierarchical Summaries
    generate_hierarchical_summaries(G)
    
    return G

def visualize_graph(G):
    """Displays the NetworkX graph with community coloring."""
    print(f"Graph Statistics: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    
    # Get partition for coloring
    undirected_G = G.to_undirected()
    partition = community_louvain.best_partition(undirected_G)
    values = [partition.get(node) for node in G.nodes()]

    # Visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    
    # Draw Nodes with community colors
    nx.draw_networkx_nodes(G, pos, node_size=2500, cmap=plt.get_cmap('viridis'), 
                           node_color=values, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw Edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Knowledge Graph Visualization (Colored by Community)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    graph_file = "knowledge_graph.graphml"

    # Check if the graph already exists
    if os.path.exists(graph_file):
        print(f"Existing graph found: '{graph_file}'. Loading for visualization and sync...")
        G = nx.read_graphml(graph_file)
        # If summaries don't exist, generate them
        if not os.path.exists("community_summaries.json"):
            generate_hierarchical_summaries(G)
        
        # Sync to Neo4j
        sync_to_neo4j(G)
        
        visualize_graph(G)
        return

    # Load source text if graph doesn't exist
    try:
        with open("source_text.txt", "r", encoding="utf-8") as f:
            source_text = f.read()
    except FileNotFoundError:
        print("Error: source_text.txt not found.")
        return

    print("No existing graph found. Extracting Knowledge Graph using Gemini...")
    try:
        kg_data = extract_kg_from_text(source_text)
        print("Successfully extracted data.")
        G = build_kg_from_data(kg_data)
        
        # Sync to Neo4j
        sync_to_neo4j(G)
        
        visualize_graph(G)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
