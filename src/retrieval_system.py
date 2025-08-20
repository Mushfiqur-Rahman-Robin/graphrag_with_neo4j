from typing import List, Optional

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from .config import config


GRAPH_RETRIEVER_AVAILABLE = False
GraphRetriever = None
Eager = None

try:
    from langchain_graph_retriever import GraphRetriever
    from graph_retriever.strategies import Eager

    GRAPH_RETRIEVER_AVAILABLE = True
except Exception:
    # If this import fails, we will fall back to vector-only retriever.
    GRAPH_RETRIEVER_AVAILABLE = False


class RetrievalSystem:
    def __init__(self, documents: List[Document]):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
        )

        # Vector store
        print("Creating vector store...")
        self.vector_store = InMemoryVectorStore.from_documents(
            documents, self.embeddings
        )
        print(f"Vector store created with {len(documents)} documents")

        # Graph retriever (preferred) or vector fallback
        self.graph_retriever = self._create_graph_retriever()

        # RAG chain
        self.rag_chain = self._create_rag_chain()

    def _create_graph_retriever(self):
        """
        Create graph retriever with Eager strategy over the vector store.
        Falls back to vector-only retriever if the package or API isn't available.
        """
        if not GRAPH_RETRIEVER_AVAILABLE:
            print(
                "⚠️ langchain-graph-retriever not available; using vector-only "
                "retriever."
            )
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.SELECT_K}
            )

        # Ensure your chunks contain these metadata keys during document processing:
        # - "source": filename
        # - "doc_id": a stable id (e.g., file stem), optionally chunk_id
        edges = [("source", "source"), ("doc_id", "doc_id")]

        # Try both known signatures for Eager across versions:
        # 1) k=...
        # 2) select_k=...
        # The rest of the arguments are stable: start_k, adjacent_k, max_depth

        def build_with_k() -> Optional[object]:
            try:
                strategy = Eager(
                    k=config.SELECT_K,
                    start_k=config.START_K,
                    adjacent_k=3,
                    max_depth=config.MAX_DEPTH,
                )
                retriever = GraphRetriever(
                    store=self.vector_store, edges=edges, strategy=strategy
                )
                print("✅ Graph retriever created successfully (k signature)")
                return retriever
            except Exception as e:
                print(f"ℹ️ Eager(k=...) failed: {e}")
                return None

        def build_with_select_k() -> Optional[object]:
            try:
                strategy = Eager(
                    select_k=config.SELECT_K,
                    start_k=config.START_K,
                    adjacent_k=3,
                    max_depth=config.MAX_DEPTH,
                )
                retriever = GraphRetriever(
                    store=self.vector_store, edges=edges, strategy=strategy
                )
                print("✅ Graph retriever created successfully "
                      "(select_k signature)")
                return retriever
            except Exception as e:
                print(f"ℹ️ Eager(select_k=...) failed: {e}")
                return None

        retriever = build_with_k()
        if retriever is None:
            retriever = build_with_select_k()

        if retriever is None:
            print(
                "⚠️ Graph retriever creation failed; falling back to "
                "vector-only retriever."
            )
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.SELECT_K}
            )

        return retriever

    def _create_rag_chain(self):
        """Create RAG chain that uses the graph-aware retriever."""

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"{doc.page_content}"
                for doc in docs
            )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context.
            Be comprehensive and cite sources when possible.

            Context:
            {context}

            Question: {question}

            Answer:
            """
        )

        chain = (
            {
                "context": self.graph_retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> str:
        """Query the system."""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def test_system(self):
        """Run some sample queries to test the system."""
        test_queries = [
            "What are the main topics discussed in the documents?",
            "Can you summarize the key concepts?",
            "What relationships exist between different entities?",
        ]

        print("\n🧪 Testing GraphRAG System...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            try:
                response = self.query(query)
                print(f"✅ Response: {response[:300]}...")
            except Exception as e:
                print(f"❌ Error: {str(e)}")