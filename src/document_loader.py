import glob
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import config

class DocumentProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.text_splitter = SemanticChunker(self.embeddings)
    
    def load_documents(self, documents_path: str = None) -> List[Document]:
        """Load and process documents from the documents directory"""
        if documents_path is None:
            documents_path = str(config.DOCUMENTS_DIR)
        
        documents = []
        docx_files = glob.glob(f"{documents_path}/*.docx")
        
        if not docx_files:
            print(f"No .docx files found in {documents_path}")
            print("Please add some .docx files to the documents folder")
            return []
        
        print(f"Found {len(docx_files)} documents to process...")
        
        for docx_file in docx_files:
            try:
                print(f"Processing: {Path(docx_file).name}")
                
                # Load document
                loader = Docx2txtLoader(docx_file)
                raw_docs = loader.load()
                
                if not raw_docs or not raw_docs[0].page_content.strip():
                    print(f"Warning: {Path(docx_file).name} appears to be empty")
                    continue
                
                # Apply semantic chunking
                chunks = self.text_splitter.split_documents(raw_docs)
                
                # Add metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source": Path(docx_file).name,
                        "chunk_id": i,
                        "doc_id": Path(docx_file).stem
                    })
                
                documents.extend(chunks)
                print(f"  Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {docx_file}: {str(e)}")
                continue
        
        print(f"Total: {len(documents)} document chunks loaded")
        return documents