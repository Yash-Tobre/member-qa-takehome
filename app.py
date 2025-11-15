import streamlit as st
from main import fetch_messages, detect_person, method_rule, method_timestamp, method_bm25, method_semantic, method_llm

st.set_page_config(page_title="Member QA", layout="wide")
st.title("Member QA — Multi-Method Chat Assistant")

# Input question
question = st.text_input("Ask a question about member messages:")

top_k = st.slider("Top K results for BM25/Semantic/LLM:", min_value=1, max_value=10, value=3)

if question:
    st.info("Fetching messages and running methods...")
    messages = fetch_messages()
    person = detect_person(question, messages)
    st.write(f"Detected person: {person if person else 'None'}")

    # Run methods
    rule_res = method_rule(question, person, messages)
    timestamp_res = method_timestamp(question, person, messages)
    try:
        bm25_res = method_bm25(question, person, messages, top_k=top_k)
    except Exception as e:
        bm25_res = {"answer": f"Error: {e}"}
    try:
        sem_res = method_semantic(question, person, messages, top_k=top_k)
    except Exception as e:
        sem_res = {"answer": f"Error: {e}"}
    try:
        llm_res = method_llm(question, person, messages, top_k=top_k)
    except Exception as e:
        llm_res = {"answer": f"Error: {e}"}

    # Display results
    st.subheader("Results by Method")
    st.markdown(f"**Rule-based:** {rule_res.get('answer')}")
    st.markdown(f"**Timestamp-aware:** {timestamp_res.get('answer')}")
    st.markdown(f"**BM25:** {bm25_res.get('answer')}")
    st.markdown(f"**Semantic:** {sem_res.get('answer')}")
    st.markdown(f"**LLM:** {llm_res.get('answer')}")

    st.markdown("---")
    st.markdown(
    """
    **Note:**  
    The first four methods — Rule-based, Timestamp-aware, BM25, and Semantic — typically return the most relevant historical message, but may be **irrelevant** to your question.  
    The LLM result is a **best-effort estimate** using those messages as context.
    """
    )

