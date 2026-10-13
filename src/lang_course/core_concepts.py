from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def demo_basic_chain():
    """Demonstrate a basic chain of LLMs."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"question": "How old are you?"})

    print(f"Result: {result}")

    return chain

def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template(
        "Translate from French to English: {text}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser

    inputs = [
        {"text": "Je suis un homme."},
        {"text": "Je suis une femme."},
        {"text": "Je suis un chat."},
    ]

    results = chain.batch(inputs)

    for source, result in zip(inputs, results):
        print(f"Result: {source['text']} => Output: {result}")

    return results

def demo_stream_execution():
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about: {topic}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser

    print("streaming...")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    demo_stream_execution()