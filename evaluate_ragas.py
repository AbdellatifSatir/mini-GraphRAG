import json
import os
import google.generativeai as genai
from config import GEMINI_MODELS_PRIORITY, retry_on_quota

# 1. Setup Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

WORKING_MODEL = None

def get_working_model():
    global WORKING_MODEL
    if WORKING_MODEL:
        return WORKING_MODEL
    for model_name in GEMINI_MODELS_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            model.generate_content("test", generation_config={"max_output_tokens": 1})
            WORKING_MODEL = model
            return WORKING_MODEL
        except:
            continue
    raise RuntimeError("All models hit quota.")

JUDGE_PROMPT = """
You are an expert RAG (Retrieval-Augmented Generation) evaluator. 
Analyze the following Question, Context, Answer, and Ground Truth.
Score the Answer based on three metrics:

1. FAITHFULNESS (0.0 to 1.0): Is every claim in the Answer supported by the Context? 1.0 means perfect support, 0.0 means complete hallucination.
2. RELEVANCE (0.0 to 1.0): Does the Answer directly address the Question? 1.0 means perfect relevance.
3. CONTEXT RECALL (0.0 to 1.0): Does the provided Context contain the information necessary to answer the Question as per the Ground Truth? 1.0 means all info is present.

Output your response in STRICT JSON format like this:
{{
    "faithfulness": 0.9,
    "relevance": 1.0,
    "context_recall": 0.8,
    "reasoning": "Brief explanation for the scores."
}}

--- DATA ---
Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}
"""

@retry_on_quota(max_retries=5, initial_wait=10)
def score_result(question, context, answer, ground_truth):
    llm = get_working_model()
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
        ground_truth=ground_truth
    )
    
    response = llm.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    try:
        return json.loads(response.text.strip())
    except:
        # Fallback if JSON fails
        return {"error": "Failed to parse judge output", "raw": response.text}

def main():
    with open("eval_results.json", "r") as f:
        results = json.load(f)

    print(f"--- Starting LLM-as-a-Judge Evaluation ({len(results)} samples) ---")
    
    final_report = []
    
    for i, res in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] Evaluating Question: {res['question']}")
        
        # Score Baseline
        print("   Scoring Baseline...")
        baseline_scores = score_result(
            res['question'], 
            "\n".join(res['baseline']['contexts']), 
            res['baseline']['answer'], 
            res['ground_truth']
        )
        
        # Score GraphRAG
        print("   Scoring GraphRAG...")
        graph_scores = score_result(
            res['question'], 
            "\n".join(res['graphrag']['contexts']), 
            res['graphrag']['answer'], 
            res['ground_truth']
        )
        
        final_report.append({
            "question": res['question'],
            "baseline": baseline_scores,
            "graphrag": graph_scores
        })

    # Summary Statistics
    def avg(key, system):
        scores = [r[system][key] for r in final_report if key in r[system]]
        return sum(scores) / len(scores) if scores else 0

    print("\n" + "="*40)
    print("📊 FINAL EVALUATION SUMMARY")
    print("="*40)
    
    for system in ["baseline", "graphrag"]:
        print(f"\n🚀 System: {system.upper()}")
        print(f"   - Faithfulness:   {avg('faithfulness', system):.2f}")
        print(f"   - Answer Relevance: {avg('relevance', system):.2f}")
        print(f"   - Context Recall:   {avg('context_recall', system):.2f}")

    with open("final_eval_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print(f"\n✅ Detailed report saved to 'final_eval_report.json'")

if __name__ == "__main__":
    main()
