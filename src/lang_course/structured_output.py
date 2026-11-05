from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def string_output():
    prompt = ChatPromptTemplate.from_template(
        "What is the capital of {country}?"
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"country": "France"})
    print(response)

def json_output():
    prompt = ChatPromptTemplate.from_template(
        "Return the capital of {country} in a JSON format."
    )

    chain = prompt | llm | JsonOutputParser()

    response = chain.invoke({"country": "France"})
    print(response)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    occupation: str = Field(description="The occupation of the person")

def pydantic_output():
    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "Extract the person from: {description}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    response = chain.invoke({"description": "John Doe is a 30-year-old software engineer."})

    print(response)

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="The review of the movie")
    rating: float = Field(description="The rating of the movie")

def pydantic_output_with_structured_output():
    structured_output = llm.with_structured_output(Movie)

    result = structured_output.invoke("Review: Inception is a great movie. Rating: 4.5")

    print(result)

def complex_pydantic_output():
    class Address(BaseModel):
        street: str = Field(description="The street of the address")
        city: str = Field(description="The city of the address")
        state: str = Field(description="The state of the address")
        zip: str = Field(description="The zip code of the address")

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        address: Address = Field(description="The address of the person")

    structured_output = llm.with_structured_output(Person)

    result = structured_output.invoke("John Doe is a 30-year-old software engineer living in 123 Main St, Anytown, USA.")

    print(result)

if __name__ == "__main__":
    #string_output()
    #json_output()
    #pydantic_output()
    #pydantic_output_with_structured_output()
    complex_pydantic_output()