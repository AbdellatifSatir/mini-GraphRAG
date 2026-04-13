import os
import re
import json
import time
import datetime
import google.generativeai as genai
from config import GEMINI_MODELS_PRIORITY, retry_on_quota
from agent_tools import GraphTools

# 1. Configuration & Setup
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

WORKING_MODEL = None
LOG_DIR = "agent_logs"

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_working_model():
    """Tries models in priority order until one works or all fail."""
    global WORKING_MODEL
    if WORKING_MODEL:
        return WORKING_MODEL

    for model_name in GEMINI_MODELS_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a tiny prompt to see if it's over quota
            model.generate_content("test")
            WORKING_MODEL = model
            print(f"Using model for Agent: {model_name}")
            return WORKING_MODEL
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"⚠️ Model {model_name} is over quota. Trying next...")
                continue
            else:
                # If it's a different error, we still try the next one but log it
                print(f"⚠️ Model {model_name} failed: {e}. Trying next...")
                continue
    
    raise RuntimeError("All configured Gemini models are over quota or failing.")

SYSTEM_PROMPT = """
You are a GraphRAG Researcher. Your goal is to answer user questions by exploring a Neo4j Knowledge Graph.

You have access to the following tools:
1. get_schema(): Returns the current graph labels and relationship types. Use this early to understand the graph structure.
2. resolve_entities(query_entities: list): Maps vague terms to exact node IDs in the graph. ALWAYS do this before querying a specific entity by ID.
3. run_cypher(cypher_query: str): Executes a query and returns the results.

--- CYPHER GUIDELINES ---
- All nodes have the label ':Entity'.
- The unique identifier for a node is 'id'.
- Most relationships are labeled ':RELATED_TO'.
- The specific meaning of a relationship is in the 'type' property (e.g., [r:RELATED_TO {type: 'founded by'}]).
- Example: MATCH (n:Entity {id: 'Apple'})-[r:RELATED_TO]->(m:Entity) RETURN m.id, r.type

--- REASONING PROTOCOL ---
You must follow this exact format for every turn:
THOUGHT: <your reasoning about what to do next>
ACTION: <TOOL_NAME>(<JSON_FORMAT_ARGUMENTS>)
STOP

Examples:
THOUGHT: I need to find information about "SpaceX". First, I'll resolve the entity name to a graph ID.
ACTION: resolve_entities(["SpaceX"])
STOP

THOUGHT: I have the schema and the resolved entity. Now I will query its relationships.
ACTION: run_cypher("MATCH (e:Entity {id: 'SpaceX'})-[r:RELATED_TO]->(target) RETURN target.id, r.type")
STOP

Once you have the final answer:
THOUGHT: I have enough information.
FINAL ANSWER: <your comprehensive answer based on the findings>

--- RULES ---
- NEVER make up facts. Only use information returned by tools.
- If a Cypher query fails, read the error message and the schema hint, then try a corrected query.
- Always use resolve_entities() first if the user mentions a specific name.
"""

def log_session_start(user_input, model_name):
    """Creates a new log file for the session."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{LOG_DIR}/session_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Agentic GraphRAG Session Log\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Question:** {user_input}\n\n")
        f.write(f"## Reasoning Trace\n\n")
    return filename

def log_turn(filename, turn, response_text, observation=None):
    """Appends a single reasoning turn to the log file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"### Turn {turn}\n")
        f.write(f"{response_text}\n\n")
        if observation:
            f.write(f"**Observation:**\n```\n{observation}\n```\n\n")

@retry_on_quota(max_retries=5, initial_wait=10)
def query_agent(user_input, model=None, tools=None):
    """Executes a single question through the agentic loop and returns answer + context."""
    if not model:
        model = get_working_model()
    if not tools:
        tools = GraphTools()
        should_close_tools = True
    else:
        should_close_tools = False

    log_file = log_session_start(user_input, model.model_name)
    chat = model.start_chat(history=[])
    prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}"
    
    final_answer = "No answer generated."
    collected_observations = []

    # Max turns for the reasoning loop
    for turn in range(15):
        try:
            print(f"\n--- Turn {turn+1} ---")
            response = chat.send_message(prompt)
            response_text = response.text
            print(response_text) # Display Thought and Action

            if "FINAL ANSWER:" in response_text:
                final_answer = response_text.split("FINAL ANSWER:")[1].strip()
                log_turn(log_file, turn + 1, response_text)
                break

            # Robust Tool Parsing
            action_match = re.search(r"ACTION:\s*(\w+)\((.*)\)", response_text, re.DOTALL | re.IGNORECASE)
            
            observation = None
            if action_match:
                tool_name = action_match.group(1).lower().strip()
                raw_args = action_match.group(2).strip()
                
                # Clean up STOP if it was captured in args
                if raw_args.endswith("STOP"):
                    raw_args = raw_args[:-4].strip()
                if raw_args.endswith(")"): # Handle extra closing paren
                     raw_args = raw_args[:-1].strip()

                print(f"🛠️  Calling Tool: {tool_name}")
                try:
                    if "get_schema" in tool_name:
                        observation = tools.get_schema()
                    elif "resolve_entities" in tool_name:
                        # Handle both ["entity"] and "entity" formats
                        if raw_args.startswith("[") and raw_args.endswith("]"):
                            entities = json.loads(raw_args.replace("'", '"'))
                        else:
                            entities = [raw_args.strip("'").strip('"')]
                        observation = tools.resolve_entities(entities)
                    elif "run_cypher" in tool_name:
                        # Strip surrounding quotes from the Cypher query
                        query = raw_args.strip().strip("'").strip('"').strip('`')
                        observation = tools.run_cypher(query)
                    else:
                        observation = f"Error: Unknown tool '{tool_name}'."
                except Exception as tool_err:
                    observation = f"Error parsing arguments for {tool_name}: {tool_err}. Use JSON format for arguments."

                print(f"👁️  Observation: {observation}")
                collected_observations.append(str(observation))
                prompt = f"OBSERVATION: {observation}"
            else:
                prompt = "OBSERVATION: Invalid format. You MUST provide 'THOUGHT:', 'ACTION: TOOL_NAME(ARGS)', and 'STOP'."
                print(f"⚠️  {prompt}")

            log_turn(log_file, turn + 1, response_text, observation)
            
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print("⚠️ Quota hit in loop, retrying...")
                time.sleep(10)
                continue
            error_msg = f"Critical Loop Error: {str(e)}."
            prompt = f"OBSERVATION: {error_msg}"
            log_turn(log_file, turn + 1, f"CRITICAL ERROR: {error_msg}")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## Final Synthesis\n")
        f.write(f"{final_answer}\n")

    if should_close_tools:
        tools.close()

    return {
        "answer": final_answer,
        "contexts": collected_observations
    }

def main():
    print("--- Welcome to the Agentic GraphRAG Assistant (Phase 10: Production) ---")
    model = get_working_model()
    tools = GraphTools()
    
    while True:
        user_input = input("\nAsk a question (or 'exit'): ").strip()
        if user_input.lower() == 'exit': break
        
        result = query_agent(user_input, model=model, tools=tools)
        print(f"\nFINAL ANSWER: {result['answer']}")

    tools.close()

if __name__ == "__main__":
    main()
