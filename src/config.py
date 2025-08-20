import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DOCUMENTS_DIR = PROJECT_ROOT / "documents"
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphragpassword")
    
    # Model Configuration
    GEMINI_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    # Graph Configuration
    MAX_DEPTH = 2
    SELECT_K = 5
    START_K = 2
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required")
        
        if not cls.DOCUMENTS_DIR.exists():
            cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        return True

config = Config()
