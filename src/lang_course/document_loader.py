import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from bs4 import BeautifulSoup

load_dotenv()

def load_text_file(file_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is sample text for the file.")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        
        for doc in documents:
            print("Document content:")
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)

    return documents

def load_web_page():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Web_scraping")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from web page")
    for doc in documents:
        print(f"Loaded {len(documents)} documents from web page")
        print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
        print(f"Preview: {documents[0].page_content[:200]}...")

if __name__ == "__main__":
    # documents = load_text_file("sample.txt")
    # print(documents)
    documents = load_web_page()
    print(documents)