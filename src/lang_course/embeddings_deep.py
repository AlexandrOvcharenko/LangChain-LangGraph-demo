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

if __name__ == "__main__":
    #basic_embedding()
    batch_embeddings()