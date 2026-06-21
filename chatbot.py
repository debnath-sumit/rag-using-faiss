from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    response = llm.invoke(question)

    print("\nBot:", response.content)
    print()