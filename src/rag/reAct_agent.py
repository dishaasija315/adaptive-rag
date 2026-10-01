"""
ReAct agent setup for document retrieval and question answering.
"""

import os

from langchain.agents import create_agent

from src.config.settings import Config
from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever


config = Config()


# --------------------------------------------------
# Initialize retriever tool
# --------------------------------------------------

tools = [get_retriever()]


# --------------------------------------------------
# Load document description if available
# --------------------------------------------------

if os.path.exists("description.txt"):
    with open("description.txt", "r", encoding="utf-8") as f:
        description = f.read()
else:
    description = None


# --------------------------------------------------
# Load system prompt
# --------------------------------------------------

system_prompt = config.prompt("system_prompt")


# --------------------------------------------------
# Create ReAct agent
# --------------------------------------------------

react_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)


# --------------------------------------------------
# Keep existing variable name
# --------------------------------------------------

agent_executor = react_agent