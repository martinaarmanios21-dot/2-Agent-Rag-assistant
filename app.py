import streamlit as st
from graph import app_graph

st.set_page_config(page_title="LangChain & Qdrant Grounded Assistant", layout="wide")
st.title("⚡ LangChain & Qdrant Grounded Q&A Assistant")
st.caption("Powered by gemini-3.6-flash, Qdrant Cloud, and LangGraph Multi-Agent Orchestration")

user_query = st.text_input("Ask a technical question about LangChain or Qdrant:", "")

if st.button("Submit Query") and user_query:
    with st.spinner("Executing Researcher and Reviewer agent loop..."):
        initial_state = {
            "question": user_query,
            "retrieved_chunks": [],
            "draft_answer": "",
            "verdict": "",
            "feedback": "",
            "attempts": 0
        }
        
        final_state = app_graph.invoke(initial_state)
        
        st.subheader("Response")
        st.write(final_state["draft_answer"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reviewer Verification Verdict")
            if final_state["verdict"] == "APPROVED":
                st.success(f"Status: {final_state['verdict']}")
            else:
                st.error(f"Status: {final_state['verdict']}")
            st.info(f"Feedback: {final_state['feedback']}")
            st.write(f"Total Handoff Loop Iterations: {final_state['attempts']}")
            
        with col2:
            st.subheader("Retrieved Citations & Context Chunks")
            for chunk in final_state["retrieved_chunks"]:
                st.code(chunk, language="text")