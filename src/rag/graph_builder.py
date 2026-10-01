"""
Graph builder module for the Adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.reAct_agent import agent_executor
from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool


config = Config()


# ==================================================
# QUERY CLASSIFIER
# ==================================================

def query_classifier(state: State):
    """
    Classify the query to determine whether it is
    related to the indexed documents.
    """

    question = state["messages"][-1].content

    # Get retriever tool
    retriever_tool = get_retriever()

    # Retrieve relevant documents
    context = retriever_tool.invoke(question)

    print("\n==============================")
    print("DOCUMENTS RECEIVED")
    print("==============================")
    print(context)

    # Structured Gemini output
    llm_with_structured_output = (
        llm.with_structured_output(RouteIdentifier)
    )

    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=[
            "question",
            "context",
        ],
    )

    chain = (
        classify_prompt
        | llm_with_structured_output
    )

    result = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    print("\n==============================")
    print("QUERY CLASSIFICATION")
    print("==============================")
    print(result.route)

    return {
        "messages": state["messages"],
        "route": result.route,
        "latest_query": question,
    }


# ==================================================
# GENERAL LLM
# ==================================================

def general_llm(state: State):
    """
    Handle general questions that are not related
    to uploaded documents.
    """

    result = llm.invoke(
        state["messages"]
    )

    print("\n==============================")
    print("GENERAL LLM")
    print("==============================")
    print(result)

    return {
        "messages": [result]
    }


# ==================================================
# RETRIEVER / REACT AGENT
# ==================================================

def retriever_node(state: State):
    """
    Retrieve information from uploaded documents
    using the ReAct agent.
    """

    query = state["latest_query"]

    print("\n==============================")
    print("RETRIEVER NODE")
    print("==============================")
    print("Query:", query)

    # --------------------------------------------------
    # IMPORTANT:
    # create_agent() expects "messages",
    # not the old {"input": query} format.
    # --------------------------------------------------

    result = agent_executor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    print("\n==============================")
    print("REACT AGENT RESULT")
    print("==============================")
    print(result)

    # --------------------------------------------------
    # Get final message from agent
    # --------------------------------------------------

    messages = result.get(
        "messages",
        []
    )

    if not messages:

        raise ValueError(
            "ReAct agent returned no messages."
        )

    final_message = messages[-1]

    # --------------------------------------------------
    # Return final message
    # --------------------------------------------------

    return {
        "messages": [
            final_message
        ]
    }


# ==================================================
# GRADE
# ==================================================

def grade(state: State):
    """
    Grade the retrieved result.
    """

    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=[
            "question",
            "context",
        ],
    )

    context = state["messages"][-1].content

    question = state["latest_query"]

    llm_with_grade = (
        llm.with_structured_output(Grade)
    )

    chain_graded = (
        grading_prompt
        | llm_with_grade
    )

    result = chain_graded.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    print("\n==============================")
    print("GRADE")
    print("==============================")
    print(result)

    return {
        "messages": state["messages"],
        "binary_score": result.binary_score,
    }


# ==================================================
# REWRITE QUERY
# ==================================================

def rewrite_query(state: State):
    """
    Rewrite the user's query for better retrieval.
    """

    query = state["latest_query"]

    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=[
            "query"
        ],
    )

    chain = (
        rewrite_prompt
        | llm
    )

    result = chain.invoke(
        {
            "query": query
        }
    )

    print("\n==============================")
    print("REWRITTEN QUERY")
    print("==============================")
    print(result.content)

    return {
        "latest_query": result.content
    }


# ==================================================
# GENERATE FINAL ANSWER
# ==================================================

def generate(state: State):
    """
    Generate the final answer using retrieved context.
    """

    context = state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=[
            "context"
        ],
    )

    generate_chain = (
        generate_prompt
        | llm
    )

    result = generate_chain.invoke(
        {
            "context": context
        }
    )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(result.content)

    return {
        "messages": [
            {
                "role": "assistant",
                "content": result.content,
            }
        ]
    }


# ==================================================
# WEB SEARCH
# ==================================================

def web_search(state: State):
    """
    Search the web for the rewritten query.
    """

    search_tool = TavilySearchResults()

    result = search_tool.invoke(
        state["latest_query"]
    )

    contents = [
        item["content"]
        for item in result
        if "content" in item
    ]

    print("\n==============================")
    print("WEB SEARCH")
    print("==============================")
    print(contents)

    return {
        "messages": [
            {
                "role": "assistant",
                "content": "\n\n".join(contents),
            }
        ]
    }


# ==================================================
# BUILD GRAPH
# ==================================================

graph = StateGraph(State)


# --------------------------------------------------
# Add nodes
# --------------------------------------------------

graph.add_node(
    "query_analysis",
    query_classifier
)

graph.add_node(
    "retriever",
    retriever_node
)

graph.add_node(
    "grade",
    grade
)

graph.add_node(
    "generate",
    generate
)

graph.add_node(
    "rewrite",
    rewrite_query
)

graph.add_node(
    "web_search",
    web_search
)

graph.add_node(
    "general_llm",
    general_llm
)


# --------------------------------------------------
# Normal edges
# --------------------------------------------------

graph.add_edge(
    START,
    "query_analysis"
)

graph.add_edge(
    "retriever",
    "grade"
)

graph.add_edge(
    "rewrite",
    "retriever"
)

graph.add_edge(
    "web_search",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

graph.add_edge(
    "general_llm",
    END
)


# --------------------------------------------------
# Conditional edges
# --------------------------------------------------

graph.add_conditional_edges(
    "query_analysis",
    routing_tool
)

graph.add_conditional_edges(
    "grade",
    doc_tool
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

builder = graph.compile()