from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pip._internal.cli import parser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Senior QA/SDET interview coach.

Help the user with:
- QA interview answers
- Selenium, Playwright, API testing
- Java, Python, Pytest
- Test case writing
- Automation framework design
- Agile testing
- CI/CD testing

Explain in simple English with practical examples.
When useful, provide interview-style answers.
"""),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

print("QA Engineer Assistant started.")
print("Type 'quit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    response = chain.invoke({
        "question": question
    })

    print("\nAssistant:")
    print(response)
    print()