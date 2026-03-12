import os
import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def build_vector_index(graph_file="knowledge_graph.graphml", index_file="node_index.faiss", mapping_file="node_mapping.pkl"):
    """Creates a FAISS index for all nodes in the Knowledge Graph."""
    if not os.path.exists(graph_file):
        print(f"Error: {graph_file} not found.")
        return

    print(f"Loading graph from {graph_file}...")
    G = nx.read_graphml(graph_file)
    nodes = list(G.nodes())
    
    if not nodes:
        print("No nodes found in the graph.")
        return

    print(f"Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(nodes)} nodes...")
    # We embed the node names. You could also concatenate node type/attributes for richer embeddings.
    embeddings = model.encode(nodes)
    
    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and node mapping
    faiss.write_index(index, index_file)
    with open(mapping_file, "wb") as f:
        pickle.dump(nodes, f)
    
    print(f"Vector index saved to {index_file}")
    print(f"Node mapping saved to {mapping_file}")

if __name__ == "__main__":
    build_vector_index()
