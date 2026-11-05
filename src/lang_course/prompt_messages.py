from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import FewShotChatMessagePromptTemplate

load_dotenv()

def translate_text(input_language, output_language, text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}"),
        ("user", "Translate the following text: {text}"),
    ])

    messages = prompt.format_messages(
        input_language=input_language,
        output_language=output_language,
        text=text,
    )

    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    response = model.invoke(messages)

    print(response.content)

    return response.content


def few_shot_prompt():
    examples = [
        {
            "input": "happy",
            "output": "sad"
        },
        {
            "input": "tall",
            "output": "short"
        },
        {
            "input": "big",
            "output": "small"
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Give me the opposite for each word"),
        few_shot_prompt,
        ("human", "{input}"),
    ])


    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    response = model.invoke(final_prompt.format_messages(input="happy"))

    print(response.content)

    return response.content

def combine_prompts():
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}")
    ])

    user_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ])

    combined_prompt = system_prompt | user_prompt

    full_prompt = system_prompt + user_prompt

    combined_prompt = combined_prompt.format_messages(role="assistant", question="What is the capital of France?")
    full_final = full_prompt.format_messages(role="assistant", question="What is the capital of France?")

    # print(combined_final)
    print(full_final)

    return full_final

if __name__ == "__main__":
    combine_prompts()