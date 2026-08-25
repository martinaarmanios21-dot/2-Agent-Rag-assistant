import json
import os
import time
from graph import app_graph

# Load questions from external JSON dataset, or default to a fallback list
EVAL_FILE = "questions.json"

if os.path.exists(EVAL_FILE):
    with open(EVAL_FILE, "r") as f:
        test_questions = json.load(f)  # Assumes a list of string questions
else:
    print(f"⚠️ {EVAL_FILE} not found. Running default test set.")
    test_questions = [
        "How do I initialize a Qdrant vector store in Python?",
        "What is the function of RecursiveCharacterTextSplitter in LangChain?",
        "Who won the 2022 FIFA World Cup?",
        "How does LangChain handle document loading?",
        "What is the capital of France?"
    ]

print(f"Starting Evaluation Suite on {len(test_questions)} Questions...\n" + "="*50)

results = []

for idx, q in enumerate(test_questions, 1):
    # Extract string if dataset uses object format [{"question": "..."}, ...]
    question_text = q["question"] if isinstance(q, dict) else q
    
    print(f"\nTest {idx}/{len(test_questions)}: {question_text}")
    
    res = app_graph.invoke({
        "question": question_text,
        "retrieved_chunks": [],
        "draft_answer": "",
        "verdict": "",
        "feedback": "",
        "attempts": 0
    })
    
    print(f"Verdict: {res['verdict']} | Attempts: {res['attempts']}")
    print(f"Final Answer: {res['draft_answer'][:120]}...")
    print("-" * 50)
    
    # Store output for metric reporting
    results.append({
        "question": question_text,
        "answer": res["draft_answer"],
        "verdict": res["verdict"]
    })
    
    # Rate-limit buffer for API calls
    time.sleep(2)

# Save evaluation report
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Benchmark complete. Saved output to eval_results.json")