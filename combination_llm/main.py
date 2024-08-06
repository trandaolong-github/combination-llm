import sys
sys.path.append("C:\Document\combination-llm")
import streamlit as st
from streamlit_chat import message

from combination_llm.components.sidebar import sidebar

from combination_llm.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from combination_llm.core.caching import bootstrap_caching

from combination_llm.core.parsing import read_file
from combination_llm.core.chunking import chunk_file
from combination_llm.core.embedding import embed_files
from combination_llm.core.qa import query_folder
from combination_llm.core.utils import get_llm, get_file_path
from combination_llm.model import M_GPT3, M_GPT4

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

EMBEDDING = "openai"
VECTOR_STORE = "faiss"

st.set_page_config(page_title="CombinationLLM", page_icon="📖", layout="wide")
st.header("CombinationLLM")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")
model = st.session_state.get("model")

if model == M_GPT3 or model == M_GPT4:
    if not openai_api_key:
        st.warning(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        if not is_open_ai_key_valid(openai_api_key, model):
            st.stop()


uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)


# with st.expander("Advanced Options"):
#     return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
#     show_full_doc = st.checkbox("Show parsed contents of the document")


if not uploaded_file:
    st.stop()


file_path = get_file_path(uploaded_file)
loader = TextLoader(file_path)
doc = loader.load()

embeddings_model = OllamaEmbeddings(model=model)

text_splitter = RecursiveCharacterTextSplitter(
            #model_name="llama3",
            #encoding_name="cl100k_base",
            chunk_size=512,
            chunk_overlap=100,
        )

chunks = text_splitter.split_documents(doc)

db = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory="./chromaDB")


retriever = db.as_retriever(search_kwargs={"k": 1})
llm = Ollama(model=model)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# print(qa.invoke("Who are you ?")["result"])

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


# try:
#     file = read_file(uploaded_file)
# except Exception as e:
#     display_file_read_error(e, file_name=uploaded_file.name)

# chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

# if not is_file_valid(file):
#     st.stop()





# with st.spinner("Indexing document... This may take a while⏳"):
#     folder_index = embed_files(
#         files=[chunked_file],
#         embedding=EMBEDDING if model != "debug" else "debug",
#         vector_store=VECTOR_STORE if model != "debug" else "debug",
#         openai_api_key=openai_api_key,
#     )

# with st.form(key="qa_form"):
#     query = st.text_area("Ask a question about the document")
#     submit = st.form_submit_button("Submit")


# if show_full_doc:
#     with st.expander("Document"):
#         # Hack to get around st.markdown rendering LaTeX
#         st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    # llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    # result = query_folder(
    #     folder_index=folder_index,
    #     query=query,
    #     return_all=return_all_chunks,
    #     llm=llm,
    # )

    answer = qa.invoke(query)
    print(answer)

####################################### REFER https://llamaindex-chat-with-docs.streamlit.app/ to create a chatbot

    with answer_col:
        st.markdown("#### Answer")
        #st.markdown(answer["result"])
        message(answer["result"])

    with sources_col:
        st.markdown("#### Sources")
        #st.markdown(answer.get("resource", "No resource found"))
        # for source in result.sources:
        #     st.markdown(source.page_content)
        #     st.markdown(source.metadata["source"])
        #     st.markdown("---")
