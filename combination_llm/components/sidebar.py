import streamlit as st

from dotenv import load_dotenv
import os

from combination_llm.model import M_GPT3, M_GPT4, M_QWEN2, M_LLAMA3

load_dotenv()

MODEL_LIST = [M_QWEN2, M_LLAMA3, M_GPT3, M_GPT4]

def sidebar():
    with st.sidebar:
        st.markdown(
            "## Choose an AI model\n"
        )

        model = st.selectbox("Model", options=MODEL_LIST)
        st.session_state["model"] = model
        if model == M_GPT3 or model == M_GPT4:
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your OpenAI API key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
                value=os.environ.get("OPENAI_API_KEY", None)
                or st.session_state.get("OPENAI_API_KEY", ""),
            )

            st.session_state["OPENAI_API_KEY"] = api_key_input

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "CombinationLLM allows you to ask questions about your "
            "documents and much more coming soon. "
        )
