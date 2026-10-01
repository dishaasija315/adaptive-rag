"""
Retriever setup and FAISS vector store configuration.
"""

import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Gemini Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


# --------------------------------------------------
# Global FAISS vector store
# --------------------------------------------------

_faiss_vectorstore = None


# --------------------------------------------------
# Create / update FAISS vector store
# --------------------------------------------------

def retriever_chain(chunks: list[Document]):
    """
    Store document chunks in FAISS using Gemini embeddings.
    """

    global _faiss_vectorstore

    try:

        if not chunks:
            print("No document chunks received.")
            return False

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        # Store globally
        _faiss_vectorstore = vectorstore

        print("FAISS vector store initialized with documents.")
        print(f"Vectorstore contains {len(chunks)} document chunks.")

        return True

    except Exception as e:

        print(f"Error storing documents in FAISS: {e}")

        return False


# --------------------------------------------------
# Get retriever tool
# --------------------------------------------------

def get_retriever():
    """
    Get a retriever connected to the FAISS vector store.
    """

    global _faiss_vectorstore

    try:

        # --------------------------------------------------
        # Existing uploaded documents
        # --------------------------------------------------

        if _faiss_vectorstore is not None:

            retriever = _faiss_vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            print(
                "Using existing FAISS vectorstore "
                "with uploaded documents."
            )

        # --------------------------------------------------
        # No documents uploaded yet
        # --------------------------------------------------

        else:

            print(
                "No documents uploaded yet, "
                "creating dummy vectorstore."
            )

            dummy_doc = Document(
                page_content=(
                    "No documents have been uploaded yet. "
                    "Please upload a document first."
                ),
                metadata={
                    "source": "initialization"
                },
            )

            _faiss_vectorstore = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings,
            )

            retriever = _faiss_vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

        # --------------------------------------------------
        # Load document description
        # --------------------------------------------------

        if os.path.exists("description.txt"):

            with open(
                "description.txt",
                "r",
                encoding="utf-8",
            ) as f:

                description = f.read().strip()

        else:

            description = (
                "the documents uploaded by the user"
            )

        # --------------------------------------------------
        # Create LangChain retriever tool
        # --------------------------------------------------

        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="retriever_customer_uploaded_documents",
            description=(
                "Use this tool to retrieve information from "
                "the documents uploaded by the user. "
                f"The documents are about: {description}. "
                "Use this tool when the user's question "
                "requires information from the uploaded documents."
            ),
        )

        return retriever_tool

    except Exception as e:

        print(
            f"Error initializing retriever: {e}"
        )

        raise