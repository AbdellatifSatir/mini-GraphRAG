import networkx as nx

def query_kg(entity_name):
    """Loads the graph and finds relationships for a specific entity."""
    try:
        # 1. Load the Graph
        G = nx.read_graphml("knowledge_graph.graphml")
    except FileNotFoundError:
        print("Error: knowledge_graph.graphml not found. Run gemini_kg_builder.py first.")
        return

    # 2. Simple Entity Search (Case-insensitive check)
    target_node = None
    for node in G.nodes():
        if entity_name.lower() in node.lower():
            target_node = node
            break

    if not target_node:
        print(f"No node found containing: '{entity_name}'")
        return

    print(f"\n--- Connections for: {target_node} ---")
    
    # 3. Find Outgoing Relationships (Successors)
    for neighbor in G.successors(target_node):
        relation = G[target_node][neighbor].get('relation', 'connected to')
        print(f" -> [{relation}] -> {neighbor}")

    # 4. Find Incoming Relationships (Predecessors)
    for neighbor in G.predecessors(target_node):
        relation = G[neighbor][target_node].get('relation', 'connected to')
        print(f" <- [{relation}] <- {neighbor}")

if __name__ == "__main__":
    print("Welcome to the Knowledge Graph Query System.")
    while True:
        user_query = input("\nEnter an entity to search (or 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            break
        query_kg(user_query)
