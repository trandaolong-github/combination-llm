import os
from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

from langchain.chat_models import ChatOpenAI
from combination_llm.core.debug import FakeChatModel
from langchain.chat_models.base import BaseChatModel


def pop_docs_upto_limit(
    query: str, chain: StuffDocumentsChain, docs: List[Document], max_len: int
) -> List[Document]:
    """Pops documents from a list until the final prompt length is less
    than the max length."""

    token_count: int = chain.prompt_length(docs, question=query)  # type: ignore

    while token_count > max_len and len(docs) > 0:
        docs.pop()
        token_count = chain.prompt_length(docs, question=query)  # type: ignore

    return docs


def get_llm(model: str, **kwargs) -> BaseChatModel:
    if model == "debug":
        return FakeChatModel()

    if "gpt" in model:
        return ChatOpenAI(model=model, **kwargs)  # type: ignore

    raise NotImplementedError(f"Model {model} not supported!")


def get_file_path(uploaded_file):
    cwd = os.getcwd()
    temp_dir = os.path.join(cwd, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
