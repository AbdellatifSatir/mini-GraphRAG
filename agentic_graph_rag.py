import os
import json
import google.generativeai as genai
from config import GEMINI_MODELS_PRIORITY
from agent_tools import GraphTools

# 1. Configuration & Setup
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

def get_working_model():
    """Tries models in priority order until one works or all fail."""
    for model_name in GEMINI_MODELS_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            # Simple test call to check quota
            model.generate_content("test", generation_config={"max_output_tokens": 1})
            print(f"Using model: {model_name}")
            return model
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

def main():
    print("--- Welcome to the Agentic GraphRAG Assistant (Phase 8) ---")
    
    try:
        model = get_working_model()
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    tools = GraphTools()
    
    while True:
        user_input = input("\nAsk a question (or 'exit'): ").strip()
        if user_input.lower() == 'exit': break

        chat = model.start_chat(history=[])
        prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}"
        
        # Max turns for the reasoning loop
        for _ in range(7):
            try:
                response = chat.send_message(prompt)
                print(f"\n{response.text}")
                
                if "FINAL ANSWER:" in response.text:
                    break
                
                # Robust Tool Parsing
                import re
                match = re.search(r"(\w+)\((.*)\)", response.text)
                if match:
                    tool_name = match.group(1).lower()
                    raw_args = match.group(2).strip()
                    
                    # Clean up arguments
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
                    
                    prompt = f"OBSERVATION: {observation}"
                    print(f"\n   -> {prompt}")
                else:
                    prompt = "OBSERVATION: No valid tool call detected. Use the format: TOOL_NAME(\"ARGUMENT\")."
                    print(f"\n   -> {prompt}")
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    print("\n⚠️ Quota hit during turn. Attempting to switch models...")
                    try:
                        model = get_working_model()
                        chat = model.start_chat(history=chat.history)
                        continue # Retry the same prompt with new model
                    except:
                        print("All models exhausted.")
                        break
                prompt = f"OBSERVATION: Error parsing action or executing tool. {str(e)}."
                print(f"\n   -> {prompt}")

    tools.close()

if __name__ == "__main__":
    main()
