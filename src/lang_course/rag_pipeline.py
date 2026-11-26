import os
import tempfile
from textwrap import dedent

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KNOWLEDGE_BASE = """
LangChain is a library for building applications that use language models.
It helps you connect a prompt, a model, a retriever, and a parser into one chain.

LangChain was created by Harrison Chase, together with other developers, in 2022.

LangGraph is a library for stateful, multi-step workflows on top of LangChain.
It is used to build agents that can branch, loop, and keep track of their state.
"""

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def create_knowledge_base():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"},
    )
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

llm = init_chat_model(
    model="gpt-6-luna",
    temperature=0.2,
    reasoning_effort="none",
    api_key=os.getenv("OPENAI_API_KEY"),
)

def demo_basic_rag():
    vector_store = create_knowledge_base()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_template(
        dedent("""
            Answer the question based only on the following context:

            {context}

            Question: {question}

            Answer:

            Make sure to answer in a concise manner,
            and if you don't know the answer, just say you don't know.
        """).strip()
    )

    def format_docs(docs) -> str:
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    for question in questions:
        response = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {response}")
        print("--------------------------------")

def demo_rag_with_sources():
    vector_store = create_knowledge_base()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )
    prompt = ChatPromptTemplate.from_template(
        dedent("""
            Answer the question based on the context below. Include which sources you used.

            Context:
            {context}

            Question: {question}

            Answer (include sources):
        """).strip()
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i + 1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
    ]

    for question in questions:
        response = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {response}")
        print("--------------------------------")

if __name__ == "__main__":
    # demo_basic_rag()
    demo_rag_with_sources()