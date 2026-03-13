import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini Configuration ---
# Priority list: The system will try these in order if a quota limit is hit.
GEMINI_MODELS_PRIORITY = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-flash-latest"
]

# Keep a default for single-model scripts
GEMINI_MODEL_NAME = GEMINI_MODELS_PRIORITY[0]

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
