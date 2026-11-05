from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = init_chat_model(
    model="gpt-6-luna",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )

    parser = StrOutputParser()
    
    chain = prompt | model | parser

    response = chain.invoke({"text": "LangChain is a library for building language model applications."})
    print(response)

def demo_parallel_chain():
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in two sentences: {text}"
    )
    keyword_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords from the following text: {text}/n Return them as a comma-separated list."
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text: {text}"
    )

    parser = StrOutputParser()

    analysis_chain = RunnableParallel(
        summary=summary_prompt | model | parser,
        keywords=keyword_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = """
        LangChain lets you turn a prompt, a model, and a parser into one chain.
        The same input can run through several chains at the same time.
        One chain writes a short summary, another pulls out the main keywords.
        A third chain reads the tone, so you can see whether the text feels positive or neutral.
    """

    response = analysis_chain.invoke({"text": text})
    print(f"Summary: {response['summary']}")
    print(f"Keywords: {response['keywords']}")
    print(f"Sentiment: {response['sentiment']}")

def demo_passthrough_chain():
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}/n"
        "Context: {context}/n/n"
        "Answer the question based on the context: {question}"
    )

    def fake_retrieval(input_dict):
        return  "LangChain was created in 2020 by Harrison Chase and other developers. It is a library for building language model applications."


    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retrieval),
            question=RunnableLambda(lambda inputs: inputs["question"]),
        )
        | RunnableLambda(
            lambda inputs: {
                "context": inputs["context"],
                "question": inputs["question"],
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({"question": "Who created LangChain?"})
    print(response)

def demo_branching_chain():
    code_prompt = ChatPromptTemplate.from_template(
        "You are coding expert. Help me with {question}"
    )

    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question: {question}"
    )

    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {question}/n Return only 'code' or 'general'."
    )

    classifier_chain = classifier_prompt | model | StrOutputParser()

    def is_code_question(input_dict):
        classification = classifier_chain.invoke(input_dict)
        print(f"Classification: {classification}")
        return classification == "code"

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser()
    )

    questions = [
        "How do I write a loop in Python?",
        "What is the capital of France?",
    ]

    for question in questions:
        response = branch.invoke({"question": question})
        print(response)

def demo_debugging_chain():
    prompt = ChatPromptTemplate.from_template(
        "Say hello to {name}"
    )

    chain = prompt | model | StrOutputParser()

    print("Chain input schema: ", chain.input_schema.model_json_schema())
    print("Chain output schema: ", chain.output_schema.model_json_schema())

    result = chain.with_config(
        run_name="hello_world",
    ).invoke({"name": "John"})

    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after model"))
        | StrOutputParser()
        | RunnableLambda(lambda x: log_step(x, "after output_parser"))
    )

    print("Debug chain execution:")
    result = debug_chain.invoke({"name": "Debug"})
    print(f"Greeting: {result}")

if __name__ == "__main__":
    demo_debugging_chain()