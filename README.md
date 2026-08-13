# Enterprise Chunking Studio

Enterprise Chunking Studio is an Agentic RAG (Retrieval-Augmented Generation) application that allows users to upload documents, extract their content, split the content into chunks, generate embeddings, store them in a FAISS vector database, retrieve relevant evidence, evaluate that evidence, and generate grounded answers using an LLM through OpenRouter.

## Features

- PDF, DOCX, TXT, Markdown, CSV and Excel document support
- Document text and table extraction
- Multiple chunking strategies
- Recursive text chunking
- Sentence Transformer embeddings
- FAISS vector database
- Semantic document search
- Agentic retrieval workflow
- Evidence evaluation
- Automatic retrieval re-planning
- OpenRouter LLM integration
- Streamlit interface
- User authentication and registration
- Agent execution trace

## Architecture

```text
User
  │
  ▼
Streamlit Application
  │
  ▼
Enterprise RAG Agent
  │
  ├── Vector Search
  ├── Document Search
  └── Table Search
  │
  ▼
Retrieved Evidence
  │
  ▼
Evidence Evaluator
  │
  ├── Sufficient ──────► Answer Generation
  │
  └── Insufficient
          │
          ▼
       Re-plan
          │
          ▼
     Another Tool


## Project Structure

```text
Enterprise-Chunking-Studio/
│
├── app.py                  # Streamlit application
├── agent.py                # Agentic RAG workflow
├── auth.py                 # User authentication
├── register.py             # User registration
│
├── config.py               # Environment and project configuration
├── extractor.py            # Document extraction
├── chunking.py             # Document chunking
├── vector_db.py            # FAISS vector database
├── llm.py                  # OpenRouter LLM integration
├── model.py                # Model-related functionality
│
├── test_agent.py           # Agent testing
├── test_chunking.py        # Chunking testing
├── test_vector_db.py       # Vector database testing
├── test_llm.py             # LLM testing
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignored files
├── .env.example            # Example environment configuration
├── README.md               # Project documentation
│
├── data/                   # Local uploaded documents
│
└── vector_store/           # Local FAISS vector database
