import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini Configuration ---
# Options: 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', etc.
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite" 

# --- Neo4j Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- File Paths ---
KNOWLEDGE_GRAPH_FILE = "knowledge_graph.graphml"
COMMUNITY_SUMMARIES_FILE = "community_summaries.json"
SOURCE_TEXT_FILE = "source_text.txt"
FAISS_INDEX_FILE = "node_index.faiss"
NODE_MAPPING_FILE = "node_mapping.pkl"
