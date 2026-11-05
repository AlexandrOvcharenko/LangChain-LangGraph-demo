from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

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

def demo_schema_inspection():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text: {text}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")

    return chain

def exercise_1():
    prompt = ChatPromptTemplate.from_template(
        "Write a short marketing tagline for the following product: {product} aimed at {audience}. Return only the tagline, no other text."
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"product": "a new type of smartphone", "audience": "tech-savvy consumers"})
    print(f"Result: {result}")
    return result

def new_way():
    model = init_chat_model(model="gpt-4o-mini", temperature=0.7, max_tokens=100)

if __name__ == "__main__":
    exercise_1()