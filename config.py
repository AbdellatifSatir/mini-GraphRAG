import os
import time
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

def retry_on_quota(max_retries=5, initial_wait=30):
    """Decorator to retry a function if it hits Gemini quota limits."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            wait = initial_wait
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "ResourceExhausted" in err_str:
                        print(f"⚠️ Quota hit. Retrying in {wait}s... (Attempt {retries+1}/{max_retries})")
                        time.sleep(wait)
                        retries += 1
                        wait *= 2  # Exponential backoff
                    else:
                        raise e
            raise RuntimeError(f"Max retries reached for function {func.__name__} due to quota limits.")
        return wrapper
    return decorator

# --- Gemini Configuration ---
# Priority list: The system will try these in order if a quota limit is hit.
GEMINI_MODELS_PRIORITY = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest"
]

# Keep a default for single-model scripts
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
