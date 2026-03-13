import os
import faiss
import pickle
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FAISS_INDEX_FILE, NODE_MAPPING_FILE
)

# Load Embedding Model for Entity Resolution
print("Loading Embedding Model for Tools...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class GraphTools:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_schema(self):
        """
        Tool: Returns the current Neo4j schema, including labels, relationship types, and specific relationship property values (like 'based in', 'founded by').
        """
        with self.driver.session() as session:
            # 1. Node Labels & Properties
            node_res = session.run("CALL db.labels()")
            labels = [r[0] for r in node_res]
            
            schema_info = "--- Neo4j Graph Schema ---\n"
            schema_info += f"Node Labels: {labels}\n\n"
            
            # 2. Relationship Types & Specific 'type' property values for RELATED_TO
            rel_res = session.run("CALL db.relationshipTypes()")
            rel_types = [r[0] for r in rel_res]
            schema_info += f"Relationship Types: {rel_types}\n"
            
            if "RELATED_TO" in rel_types:
                # Find the distinct 'type' property values for RELATED_TO relationships
                type_res = session.run("MATCH ()-[r:RELATED_TO]->() RETURN DISTINCT r.type AS rel_property_type")
                types = [r["rel_property_type"] for r in type_res]
                schema_info += f"Common 'RELATED_TO' relationship properties (r.type): {types}\n"
            
            return schema_info

    def resolve_entities(self, query_entities):
        """
        Tool: Maps vague entity names from the query to exact node IDs in the graph using FAISS.
        Example: 'Sarah' -> 'Sarah Chen'
        """
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(NODE_MAPPING_FILE):
            return "Error: FAISS index or mapping file missing."

        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(NODE_MAPPING_FILE, "rb") as f:
            all_nodes = pickle.load(f)

        resolved = {}
        for entity in query_entities:
            query_embedding = embed_model.encode([entity]).astype('float32')
            distances, indices = index.search(query_embedding, k=2)
            candidates = [all_nodes[idx] for idx in indices[0] if idx != -1]
            resolved[entity] = candidates
        
        return f"Entity Resolution Results: {resolved}"

    def run_cypher(self, cypher_query):
        """
        Tool: Executes a Cypher query on the Neo4j database and returns the results.
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                data = [record.data() for record in result]
                return f"Cypher Results: {data}" if data else "Cypher Results: No data found."
        except Exception as e:
            return f"Error executing Cypher: {str(e)}"
