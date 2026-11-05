from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable, Client

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    print(f"LangSmith is configured. Project name: {os.environ.get('LANGSMITH_PROJECT')}")

class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="Confidence level of the answer: high, medium, low")
    reasoning: str = Field(description="The reasoning process used to answer the question")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions to the answer",
        default_factory=list
    )
    sources_needed: bool = Field(
        description="Whether the answer requires additional sources",
        default=False
    )

class SmartQABot: 
    def __init__(
        self,
        model_name: str = "gpt-6-luna",
        temperature: float = 0.3
    ):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            reasoning_effort="none",
        ).with_structured_output(QAResponse)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """You are a helpful Q&A assistant that can answer questions and follow up questions.
                    Your guidelines:
                    - Always answer the question truthfully and accurately.
                    - If you don't know the answer, say so.
                    - If you are unsure about the answer, say so.
                    - Be honest about your confidence in the answer - set confidence level to high, medium, or low.
                    - Suggest follow-up questions if the answer is not complete.
                    - Indicate if the answer requires additional sources to be answered."""),
            (
                "human",
                "{question}"
            )
        ])

        self.chain = self.prompt | self.model

    @traceable(name="answer_question", run_type="llm", client=Client())
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return QAResponse(
                answer="Sorry, I encountered an error while processing your request.",
                confidence="low",
                reasoning="The error was: " + str(e),
                follow_up_questions=["Please try again later."],
                sources_needed=False
            )

    @traceable(name="ask_batch", run_type="llm", client=Client())
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)

def demo_qa_bot():
    bot = SmartQABot()
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]
    responses = []
    for question in questions:
        response = bot.ask(question)
        responses.append(response)
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("--------------------------------")
    print(responses)

if __name__ == "__main__":
    try:
        demo_qa_bot()
    finally:
        Client().flush()