"""
Gemini LLM initialization and configuration.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0
)