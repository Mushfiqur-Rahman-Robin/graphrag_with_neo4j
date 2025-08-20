from typing import List
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from .config import config

class GraphProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0
        )
        
        # Initialize Graph Transformer with current API
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Person", "Organization", "Location", "Concept", "Technology", "Project"],
            allowed_relationships=["WORKS_WITH", "LOCATED_IN", "MANAGES", "IMPLEMENTS", "RELATED_TO", "PART_OF"],
            node_properties=False,
            relationship_properties=False
        )
        
        # Initialize Neo4j connection
        self.neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD
        )
    
    def create_knowledge_graph(self, documents: List[Document]) -> bool:
        """Create knowledge graph from documents"""
        try:
            print("Converting documents to graph format...")
            
            # Convert documents to graph documents
            graph_documents = self.graph_transformer.convert_to_graph_documents(documents)
            print(f"Created {len(graph_documents)} graph documents")
            
            # Store in Neo4j
            print("Storing graph in Neo4j...")
            self.neo4j_graph.add_graph_documents(graph_documents)
            
            # Verify storage
            result = self.neo4j_graph.query("MATCH (n) RETURN count(n) as node_count")
            node_count = result[0]["node_count"] if result else 0
            
            result = self.neo4j_graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result[0]["rel_count"] if result else 0
            
            print(f"Graph created successfully: {node_count} nodes, {rel_count} relationships")
            return True
            
        except Exception as e:
            print(f"Error creating knowledge graph: {str(e)}")
            return False
    
    def clear_graph(self):
        """Clear the existing graph"""
        try:
            self.neo4j_graph.query("MATCH (n) DETACH DELETE n")
            print("Graph cleared successfully")
        except Exception as e:
            print(f"Error clearing graph: {str(e)}")
    
    def get_graph_stats(self) -> dict:
        """Get graph statistics"""
        try:
            nodes_result = self.neo4j_graph.query("MATCH (n) RETURN count(n) as count")
            rels_result = self.neo4j_graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            
            return {
                "nodes": nodes_result[0]["count"] if nodes_result else 0,
                "relationships": rels_result[0]["count"] if rels_result else 0
            }
        except Exception as e:
            print(f"Error getting graph stats: {str(e)}")
            return {"nodes": 0, "relationships": 0}