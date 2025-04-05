"""
Creating chatbots
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and knowledgeable medical assistant. Your task is to provide accurate and context-aware answers based on the provided medical literature.

    - Always use the provided **context** to answer the user's **question**.
    - Be clear, concise, and professional in your responses.
    - Do not make up information or guess beyond the given context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

chain = prompt | model

def get_bot_response(user_ques:str) -> str:
    context = retriever.invoke(user_ques)
    result = chain.invoke({
        "context" : context , 
        "question" : user_ques
    })

    return result.content
