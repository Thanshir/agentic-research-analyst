"""
Simple Streamlit UI for demoing the agent.
Run with: streamlit run ui.py
"""
import streamlit as st
from dotenv import load_dotenv

from src.graph import run_agent

load_dotenv()

st.set_page_config(page_title="Agentic Research Analyst", page_icon="🔎")
st.title("🔎 Agentic Research Analyst")
st.caption("A self-reflecting LangGraph agent — plans, searches, critiques its own answer, and retries if needed.")

question = st.text_area(
    "Ask a research question",
    placeholder="e.g. Compare AWS EC2 vs GCP Compute Engine pricing for a small production workload",
    height=100,
)

if st.button("Run Research", type="primary") and question.strip():
    with st.spinner("Agent is researching (this may loop a few times if it self-critiques)..."):
        try:
            result = run_agent(question)

            st.subheader("Answer")
            st.write(result.answer.replace("$", "\\$"))

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{result.confidence:.0%}")
            col2.metric("Needs More Research", "Yes" if result.needs_more_research else "No")

            with st.expander("Reasoning"):
                st.write(result.reasoning.replace("$", "\\$"))

            if result.sources:
                st.subheader("Sources")
                for s in result.sources:
                    if s.url:
                        st.markdown(f"- [{s.title}]({s.url})")
                    else:
                        st.markdown(f"- {s.title}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Tip: check your LangSmith dashboard to see the full trace of tool calls and reflection loops for this run.")
