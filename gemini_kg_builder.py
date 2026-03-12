import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
from community import community_louvain
from dotenv import load_dotenv

# 1. Configuration & Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please check .env.example.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

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

def generate_community_summaries(G):
    """Groups nodes into communities and asks Gemini to summarize each."""
    # Louvain works on undirected graphs
    undirected_G = G.to_undirected()
    partition = community_louvain.best_partition(undirected_G)
    
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    
    summaries = {}
    print(f"Detected {len(communities)} communities. Summarizing...")
    
    for comm_id, nodes in communities.items():
        # Get edges within this community for context
        edges = []
        for u, v, d in G.edges(data=True):
            if u in nodes and v in nodes:
                edges.append(f"{u} --[{d['relation']}]--> {v}")
        
        prompt = f"""
        Given this community of entities and their relationships, provide a one-sentence summary of what this community represents in the context of the larger document.
        
        Entities: {', '.join(nodes)}
        Relationships: {', '.join(edges)}
        
        Summary:"""
        
        response = model.generate_content(prompt)
        summaries[comm_id] = {
            "nodes": nodes,
            "summary": response.text.strip()
        }
        print(f"   Community {comm_id}: {summaries[comm_id]['summary'][:50]}...")
    
    # Save summaries to JSON
    with open("community_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4)
    print("Community summaries saved to 'community_summaries.json'.")

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
    
    # NEW: Generate Community Summaries
    generate_community_summaries(G)
    
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
        print(f"Existing graph found: '{graph_file}'. Loading for visualization...")
        G = nx.read_graphml(graph_file)
        # If summaries don't exist, generate them
        if not os.path.exists("community_summaries.json"):
            generate_community_summaries(G)
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
        visualize_graph(G)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
