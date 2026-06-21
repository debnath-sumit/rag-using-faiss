from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Senior QA/SDET interview coach.
Explain in simple English with practical examples.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

chat_history = []

print("QA Assistant with Memory started. Type 'quit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "question": question
        })

        print("\nAssistant:")
        print(response)
        print()

        chat_history.append(("human", question))
        chat_history.append(("ai", response))

    except Exception as e:
        print("\nAssistant:")
        print("Gemini quota limit reached. Please try again after some time or switch to gemini-2.5-flash-lite.")
        print()