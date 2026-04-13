import os
import faiss
import pickle
import threading
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FAISS_INDEX_FILE, NODE_MAPPING_FILE
)

# --- Performance Caching (Phase 10) ---
_embed_model = None
_model_lock = threading.Lock()

def get_embed_model():
    """Singleton pattern for loading the embedding model."""
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            print("Loading Embedding Model for Tools...")
            _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

class GraphTools:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.resolution_cache = {} # Cache for resolve_entities
        self._schema_cache = None  # Cache for get_schema
        
        # Pre-load Vector Index for fast Entity Resolution
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(NODE_MAPPING_FILE):
            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                with open(NODE_MAPPING_FILE, "rb") as f:
                    self.all_nodes = pickle.load(f)
                print("✅ Vector Index and Node Mapping loaded.")
            except Exception as e:
                print(f"⚠️ Error loading Vector Index: {e}")
                self.index = None
                self.all_nodes = None
        else:
            print("⚠️ Vector Index files missing. Entity resolution will be disabled.")
            self.index = None
            self.all_nodes = None

    def close(self):
        self.driver.close()

    def get_schema(self):
        """
        Tool: Returns the current Neo4j schema, including labels, relationship types, and specific relationship property values.
        Uses caching for performance.
        """
        if self._schema_cache:
            return self._schema_cache

        try:
            with self.driver.session() as session:
                # 1. Node Labels
                node_res = session.run("CALL db.labels()")
                labels = [r[0] for r in node_res]
                
                schema_info = "--- Neo4j Graph Schema ---\n"
                schema_info += f"Node Labels: {labels}\n\n"
                
                # 2. Relationship Types
                rel_res = session.run("CALL db.relationshipTypes()")
                rel_types = [r[0] for r in rel_res]
                schema_info += f"Relationship Types: {rel_types}\n"
                
                if "RELATED_TO" in rel_types:
                    type_res = session.run("MATCH ()-[r:RELATED_TO]->() RETURN DISTINCT r.type AS rel_property_type")
                    types = [r["rel_property_type"] for r in type_res]
                    schema_info += f"Common 'RELATED_TO' relationship properties (r.type): {types}\n"
                
                self._schema_cache = schema_info
                return schema_info
        except Exception as e:
            return f"Error retrieving schema: {e}"

    def resolve_entities(self, query_entities):
        """
        Tool: Maps vague entity names from the query to exact node IDs using FAISS with result caching.
        """
        if self.index is None or self.all_nodes is None:
            return "Error: Vector Index not available for resolution."

        embed_model = get_embed_model()
        resolved = {}
        for entity in query_entities:
            entity = entity.strip()
            # Check cache first
            if entity in self.resolution_cache:
                resolved[entity] = self.resolution_cache[entity]
                continue

            try:
                query_embedding = embed_model.encode([entity]).astype('float32')
                distances, indices = self.index.search(query_embedding, k=2)
                candidates = [self.all_nodes[idx] for idx in indices[0] if idx != -1]
                resolved[entity] = candidates
                self.resolution_cache[entity] = candidates # Save to cache
            except Exception as e:
                resolved[entity] = f"Error during resolution: {str(e)}"
        
        return f"Entity Resolution Results: {resolved}"

    def run_cypher(self, cypher_query):
        """
        Tool: Executes a Cypher query with intelligent self-correction hints.
        """
        # Minor pre-cleaning: Strip leading/trailing quotes often added by LLMs
        cypher_query = cypher_query.strip().strip("'").strip('"').strip('`')
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                data = [record.data() for record in result]
                return f"Cypher Results: {data}" if data else "Cypher Results: No data found."
        except Exception as e:
            # --- PHASE 10: Automatic Error Analysis ---
            error_msg = str(e)
            schema = self.get_schema()
            
            # Simple self-correction suggestions
            correction_hint = ""
            if "not found" in error_msg.lower() or "PropertyNotFound" in error_msg:
                correction_hint = "\n💡 TIP: Check if the property name exists in the schema. Remember to use 'r.type' for relationship properties."
            elif "LabelNotFound" in error_msg:
                correction_hint = f"\n💡 TIP: Check node labels. Available: {schema.split('Node Labels: ')[1].split('\\n')[0]}"
            elif "SyntaxError" in error_msg or "Invalid input" in error_msg:
                correction_hint = "\n💡 TIP: Check for missing brackets, quotes, or keywords. Use (n:Entity) for nodes."

            return f"❌ Cypher Error: {error_msg}{correction_hint}\n\nRELEVANT SCHEMA:\n{schema}"
