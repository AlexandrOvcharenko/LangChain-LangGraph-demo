from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain_core")
openai_version = version("langchain_openai")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"Langchain Core version: {core_version}")
print(f"Langgraph version: {lg_version}")

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    response = llm.invoke("What is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
