# GraphRAG System with LangChain, Gemini, and Neo4j

A production-ready GraphRAG (Graph-Augmented Retrieval) system that combines vector search with knowledge graphs for superior document retrieval and question answering.

## Features

- **Document Processing**: Semantic chunking of .docx files
- **Knowledge Graph Creation**: Automated entity and relationship extraction
- **Hybrid Retrieval**: Combines vector similarity with graph traversal
- **Interactive Notebooks**: Step-by-step implementation and testing
- **Docker Integration**: Easy Neo4j setup and deployment

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd graphrag_with_neo4j
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Google API key
```

### 4. Start Neo4j

```bash
docker compose up
```

### 5. Add Documents

Place your .docx files in the `documents/` folder.

### 6. Run the System

Open and run the notebooks in order:
1. `01_setup_and_test.ipynb` - Verify setup
2. `02_document_processing.ipynb` - Process documents
3. `03_graph_creation.ipynb` - Create knowledge graph
4. `04_retrieval_testing.ipynb` - Test retrieval
5. `05_complete_system.ipynb` - Full pipeline

## System Architecture