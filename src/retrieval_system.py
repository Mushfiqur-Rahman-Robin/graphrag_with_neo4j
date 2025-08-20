from typing import List

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from .config import config

# Try to import GraphRetriever + Eager if available
try:
    from langchain_graph_retriever import GraphRetriever, Eager
except ImportError:
    GraphRetriever = None
    Eager = None


class RetrievalSystem:
    def __init__(self, documents: List[Document]):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
        )

        # Create vector store
        print("Creating vector store...")
        self.vector_store = InMemoryVectorStore.from_documents(
            documents, self.embeddings
        )
        print(f"Vector store created with {len(documents)} documents")

        # Create retriever (graph if available, else vector fallback)
        self.graph_retriever = self._create_graph_retriever()

        # Create RAG chain
        self.rag_chain = self._create_rag_chain()

    def _create_graph_retriever(self):
        """Create graph retriever if available, else fall back to vector retriever."""
        if GraphRetriever is None or Eager is None:
            print(
                "⚠️ langchain-graph-retriever not available; falling back to vector-only retriever."
            )
            return self.vector_store.as_retriever(search_kwargs={"k": config.SELECT_K})

        try:
            strategy = Eager(
                select_k=config.SELECT_K,
                start_k=config.START_K,
                adjacent_k=3,
                max_depth=config.MAX_DEPTH,
            )

            graph_retriever = GraphRetriever(
                store=self.vector_store,
                edges=[("source", "source"), ("doc_id", "doc_id")],
                strategy=strategy,
            )

            print("✅ Graph retriever created successfully")
            return graph_retriever

        except Exception as e:
            print(f"⚠️ Graph retriever creation failed: {str(e)}")
            print("➡️ Falling back to standard vector retriever")
            return self.vector_store.as_retriever(
                search_kwargs={"k": config.SELECT_K}
            )

    def _create_rag_chain(self):
        """Create RAG chain"""

        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
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
        """Query the system"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def test_system(self):
        """Run some sample queries to test the system"""
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
