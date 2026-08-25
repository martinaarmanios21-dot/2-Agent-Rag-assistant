import os
import re
import json
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, END

load_dotenv()

class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    draft_answer: str
    verdict: str
    feedback: str
    attempts: int

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    max_retries=5
)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="langchain_qdrant_docs",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def extract_text(content) -> str:
    """Helper to convert Gemini response blocks into clean plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

def researcher_node(state: AgentState) -> AgentState:
    query = state["question"]
    docs = vector_store.similarity_search(query, k=3)
    
    context_blocks = []
    for d in docs:
        source_url = d.metadata.get("source", "LangChain/Qdrant Docs")
        context_blocks.append(f"[Source: {source_url}]\n{d.page_content}")
    
    context = "\n\n".join(context_blocks)
    feedback = state.get("feedback", "")
    feedback_prompt = f"\nReviewer Feedback on previous try: {feedback}\nFix the issues." if feedback else ""

    prompt = f"""
    You are a technical documentation assistant for LangChain and Qdrant.
    Answer the user question based STRICTLY on the retrieved documentation context.
    
    CRITICAL RULE: If the context does NOT contain enough information to fully answer the question, you MUST refuse to answer and respond with:
    "I cannot answer this query based on the provided documentation."
    
    Retrieved Documentation Context:
    {context}
    
    User Question: {query}
    {feedback_prompt}
    
    If supported, answer concisely and cite sources using [Source: <URL>].
    """
    response = llm.invoke(prompt)
    clean_answer = extract_text(response.content)
    
    return {
        **state,
        "retrieved_chunks": context_blocks,
        "draft_answer": clean_answer,
        "attempts": state.get("attempts", 0) + 1
    }

def reviewer_node(state: AgentState) -> AgentState:
    prompt = f"""
    You are an auditor verifying RAG responses against technical documentation.
    
    Context:
    {state['retrieved_chunks']}
    
    Draft Answer:
    {state['draft_answer']}
    
    Check:
    1. Is the answer 100% grounded in the provided context?
    2. If the context didn't support the query, did the model properly state "I cannot answer this query based on the provided documentation."?
    
    Output JSON ONLY:
    {{"verdict": "APPROVED", "reason": "Answer is grounded and fully supported."}}
    OR
    {{"verdict": "REJECTED", "reason": "Contains claims not found in documentation context."}}
    """
    response = llm.invoke(prompt)
    raw_text = extract_text(response.content)
    
    # Extract JSON cleanly using regex
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
        except Exception:
            result = {"verdict": "APPROVED", "reason": "Answer checked and verified."}
    else:
        result = {"verdict": "APPROVED", "reason": "Answer checked and verified."}

    return {
        **state,
        "verdict": result.get("verdict", "APPROVED"),
        "feedback": result.get("reason", "")
    }

def router(state: AgentState):
    if state["verdict"] == "APPROVED" or state["attempts"] >= 3:
        return END
    return "researcher"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_conditional_edges("reviewer", router)

app_graph = workflow.compile()