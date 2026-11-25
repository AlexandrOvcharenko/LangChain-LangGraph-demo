from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile

from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

SAMPLE_DOCUMENTS = [
    Document(
        page_content="Java is a programming language used to build applications.",
        metadata={"source": "java.txt"},
    ),
    Document(
        page_content="Java is a popular programming language for writing application software.",
        metadata={"source": "java_again.txt"},
    ),
    Document(
        page_content="Java is a programming language that developers use to create applications.",
        metadata={"source": "java_once_more.txt"},
    ),
    Document(
        page_content="Java is an island in Indonesia, known for volcanoes and coffee.",
        metadata={"source": "java_island.txt"},
    ),
    Document(
        page_content="Cats are popular pets that often sleep during the day.",
        metadata={"source": "cats.txt"},
    ),
]

def chroma_basics():
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCUMENTS,
            embedding=embeddings_model,
            persist_directory=temp_dir,
        )

        query = "What is java?"
        results = vectorstore.similarity_search_with_score(query, k=2)
        print(f"Query: {query}")
        print("Ranked documents:")
        for doc, distance in results:
            print(f"Distance: {distance:.4f}: {doc.page_content}")
            print("-" * 50)

def persist_chrome():
    persist_directory = "chroma_db"

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCUMENTS,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )

    original_count = vectorstore._collection.count()
    print(f"Original count: {original_count}")

   # simmulate restart

    del vectorstore

    reloaded_vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_model,
    )

    reloaded_count = reloaded_vectorstore._collection.count()
    print(f"Reloaded count: {reloaded_count}")

    return reloaded_vectorstore

def as_retriever():
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCUMENTS,
            embedding=embeddings_model,
            persist_directory=temp_dir,
        )
        query = "What is Java?"
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        print(f"Query: {query}")
        print("similarity:")
        for doc in retriever.invoke(query):
            print(f"- {doc.metadata['source']}: {doc.page_content}")

        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.6},
        )
        print("mmr:")
        for doc in mmr_retriever.invoke(query):
            print(f"- {doc.metadata['source']}: {doc.page_content}")

def exercise_vector_store_setup():

    def create_retriever(
        texts: list[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 3,
    ):

        docs = [
            Document(page_content=text) for text in texts
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model
        )

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        
    sample_texts = [
        """Harbor Lane Bakery staff note.

Mira Solano opened the bakery on 14 March 2019 at 18 Candlewick Alley. The shop bell is from her grandmother's kitchen in Odesa.

The rye starter is named Bramble. Feed Bramble every 22 hours with 40 g of rye flour and 40 g of water. If Bramble smells like nail polish, throw it out and restart from the jar labeled March.

Saturday wholesale pickup uses the code HARBOR-77. Say it at the side door after 6:10, not at the counter.

The spare key for the oven room is taped inside the blue flour tin on the top shelf. Do not leave it in the till.

The pistachio roll uses almond flour. It is not safe for guests who asked to avoid almonds, even if they only mentioned peanuts.
""",
        "The night tram on route 14 in Pell stops at Glassmarket only when a yellow scarf is lying on the bench.",
        "Locker C-19 at Vistula Records opens with 4418. It holds rehearsal tapes from 1998 and nothing else.",
    ]

    query = "How often should Bramble the rye starter be fed, and with how much flour?"
    retriever = create_retriever(
        sample_texts,
        chunk_size=240,
        chunk_overlap=40,
        k=2,
    )
    print(f"Query: {query}")
    for doc in retriever.invoke(query):
        print(f"- {doc.page_content}")
        print("-" * 40)


if __name__ == "__main__":  
    #as_retriever()
    #persist_chrome()
    exercise_vector_store_setup()