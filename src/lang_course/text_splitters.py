from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SAMPLE_TEXT = """
This is a sample text for the text splitter.
It is used to test the text splitter.
It is used to test the text splitter.
"""

SAMPLE_CODE = """
import os
import tempfile

def load_text_file(file_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is sample text for the file.").read().encode("utf-8"))
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

if __name__ == "__main__":
    documents = load_text_file(SAMPLE_TEXT)
    print(documents)
    documents = load_text_file(SAMPLE_CODE)
    print(documents)
"""

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Split {len(chunks)} chunks")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Preview: {chunks[0][:100]}...")

if __name__ == "__main__":
    recursive_splitter()