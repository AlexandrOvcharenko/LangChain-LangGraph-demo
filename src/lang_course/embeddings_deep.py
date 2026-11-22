from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def basic_embedding():
    text = "What is machine learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimension: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    texts = ["What is machine learning?", "What is deep learning?", "What is artificial intelligence?"]
    batch_embeddings = embeddings.embed_documents(texts)

    for i, embedding in enumerate(batch_embeddings):
        print(f"Text: {texts[i]}")
        print(f"Vector dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Vector norm: {np.linalg.norm(embedding):.4f}")
        print("-" * 50)

def similarity_search():
    docs = [
        "Python is a programming language.",
        "Java is a programming language.",
        "Deep learning is a type of machine learning.",
        "Cats are popular pets.",
    ]

    query = "What is a programming language?"

    doc_vectors = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(doc_vector, query_vector) for doc_vector in doc_vectors]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("Ranked documents:")
    for doc, similarity in ranked_docs:
        print(f"Similarity: {similarity:.4f}: {doc}")


if __name__ == "__main__":
    #basic_embedding()
    #batch_embeddings()
    similarity_search()