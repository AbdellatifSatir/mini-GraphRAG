import os
import json
import google.generativeai as genai
from config import GEMINI_MODELS_PRIORITY, retry_on_quota
from agent_tools import GraphTools

# 1. Configuration & Setup
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

WORKING_MODEL = None

def get_working_model():
    """Tries models in priority order until one works or all fail."""
    global WORKING_MODEL
    if WORKING_MODEL:
        return WORKING_MODEL

    for model_name in GEMINI_MODELS_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            # Simple test call to check quota
            model.generate_content("test", generation_config={"max_output_tokens": 1})
            print(f"Using model for Agent: {model_name}")
            WORKING_MODEL = model
            return WORKING_MODEL
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"⚠️ Quota hit for {model_name}. Trying next...")
                continue
            else:
                print(f"⚠️ Error with {model_name}: {e}. Trying next...")
                continue
    raise RuntimeError("All Gemini models in priority list failed or hit quota.")

SYSTEM_PROMPT = """
You are a GraphRAG Researcher. Your goal is to answer user questions by exploring a Neo4j Knowledge Graph.

You have access to the following tools:
1. get_schema(): Returns the current graph labels and relationship types.
2. resolve_entities(query_entities: list): Maps vague terms to exact node IDs in the graph.
3. run_cypher(cypher_query: str): Executes a query and returns the results.

--- CYPHER TIPS ---
- Most relationships are labeled :RELATED_TO. 
- Specific meanings are stored in the 'type' property of the relationship.
- Example to find who founded a company:
  MATCH (c:Entity)-[r:RELATED_TO {type: 'founded by'}]->(founder:Entity) RETURN founder.id
- Example to find a company based in Berlin:
  MATCH (c:Entity)-[r:RELATED_TO {type: 'based in'}]->(l:Entity {id: 'Berlin'}) RETURN c.id

--- REASONING PROTOCOL ---
1. THOUGHT: Explain what you need to find.
2. ACTION: Call ONE tool in the format: TOOL_NAME("ARGUMENT")
3. STOP: Stop writing after the ACTION.
4. OBSERVATION: Real data from the system.
5. FINAL ANSWER: Summarize findings.

--- CRITICAL RULES ---
- Never make up data.
- Always use resolve_entities() to get the correct 'id' of a node before querying it in Cypher.
- If a tool fails, try a different Cypher query.
"""

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

    chat = model.start_chat(history=[])
    prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}"
    
    final_answer = "No answer generated."
    collected_observations = []

    # Max turns for the reasoning loop
    for _ in range(7):
        try:
            response = chat.send_message(prompt)
            
            if "FINAL ANSWER:" in response.text:
                final_answer = response.text.split("FINAL ANSWER:")[1].strip()
                break
            
            # Robust Tool Parsing
            import re
            match = re.search(r"(\w+)\((.*)\)", response.text)
            if match:
                tool_name = match.group(1).lower()
                raw_args = match.group(2).strip()
                
                if "=" in raw_args and not (raw_args.startswith("{") or raw_args.startswith("[")):
                    raw_args = raw_args.split("=", 1)[1].strip()
                
                if "get_schema" in tool_name:
                    observation = tools.get_schema()
                elif "resolve_entities" in tool_name:
                    clean_args = raw_args.strip("[]").split(",")
                    entities = [e.strip().strip("'").strip('"') for e in clean_args]
                    observation = tools.resolve_entities(entities)
                elif "run_cypher" in tool_name:
                    observation = tools.run_cypher(raw_args.strip("'").strip('"'))
                else:
                    observation = f"Error: Unknown tool '{tool_name}'."
                
                collected_observations.append(str(observation))
                prompt = f"OBSERVATION: {observation}"
            else:
                prompt = "OBSERVATION: No valid tool call detected. Use the format: TOOL_NAME(\"ARGUMENT\")."
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                model = get_working_model()
                chat = model.start_chat(history=chat.history)
                continue
            prompt = f"OBSERVATION: Error. {str(e)}."

    if should_close_tools:
        tools.close()

    return {
        "answer": final_answer,
        "contexts": collected_observations
    }

def main():
    print("--- Welcome to the Agentic GraphRAG Assistant (Phase 8) ---")
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
