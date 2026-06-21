import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python teacher."),
    ("human", "Explain this topic in simple English: {topic}")
])

chain = prompt | llm

topic = input("Enter a topic: ")

response = chain.invoke({
    "topic": topic
})

print("\n")
print(response.content)
