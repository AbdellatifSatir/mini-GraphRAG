import json
import os
from baseline_rag import baseline_rag_query
from agentic_graph_rag import query_agent, get_working_model
from agent_tools import GraphTools

def main():
    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    results = []
    
    print("Starting Data Collection for Evaluation...")
    
    # Initialize Agent components once to save time/quota
    model = get_working_model()
    tools = GraphTools()

    for i, item in enumerate(dataset):
        question = item["question"]
        print(f"\n[{i+1}/{len(dataset)}] Question: {question}")

        # 1. Run Baseline RAG
        print("   Running Baseline RAG...")
        baseline_res = baseline_rag_query(question)
        
        # 2. Run GraphRAG Agent
        print("   Running GraphRAG Agent...")
        graph_res = query_agent(question, model=model, tools=tools)

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "baseline": {
                "answer": baseline_res["answer"],
                "contexts": baseline_res["contexts"]
            },
            "graphrag": {
                "answer": graph_res["answer"],
                "contexts": graph_res["contexts"]
            }
        })

        # if i == 0:
        #     break  # Remove this line to run on the full dataset

    tools.close()

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Evaluation results saved to 'eval_results.json'")

if __name__ == "__main__":
    main()
