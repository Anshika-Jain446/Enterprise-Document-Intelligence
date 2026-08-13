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


##Requirements
Python 3.11+
OpenRouter API key
Internet connection
Required Python packages from requirements.txt

##Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

##Install dependencies:

pip install -r requirements.txt

Environment Configuration

Create a .env file in the project root.

OPENROUTER_API_KEY=your_openrouter_api_key_here

LLM_MODEL=meta-llama/llama-3.3-70b-instruct

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTOR_DB_PATH=vector_store

CHUNK_SIZE=1000

CHUNK_OVERLAP=200

The actual .env file must NOT be uploaded to GitHub.

Only provide an example configuration such as .env.example.

Running the Application

Start the Streamlit application:

streamlit run app.py

Then open the local Streamlit URL displayed in the terminal.

Testing
Test LLM
python test_llm.py
Test Chunking
python test_chunking.py
Test Vector Database
python test_vector_db.py
Test Agent
python test_agent.py
Agent Workflow

The agent receives a user question and decides which retrieval tool should be used.

User Question
      │
      ▼
Agent Planner
      │
      ├── Vector Search
      │
      ├── Document Search
      │
      └── Table Search
      │
      ▼
Retrieve Evidence
      │
      ▼
Evaluate Evidence
      │
      ├── Sufficient
      │      │
      │      ▼
      │   Generate Answer
      │
      └── Insufficient
             │
             ▼
           Re-plan
             │
             ▼
       Try Another Tool

The agent also maintains a trace of its decisions, retrieved results, evidence evaluation, and final answer generation.

Vector Database

The project uses:

Sentence Transformers
        │
        ▼
   Text Embeddings
        │
        ▼
       FAISS
        │
        ▼
Semantic Search

The vector database stores document chunks and their corresponding embeddings so that relevant information can be retrieved based on semantic similarity.

LLM

The application uses OpenRouter as the LLM provider.

The LLM is responsible for:

Agent planning
Evidence evaluation
Grounded answer generation

The model is configured through the LLM_MODEL environment variable.

Security

Never commit secrets to GitHub.

Do NOT upload:

.env
users.json
vector_store/
venv/

Do not place API keys directly inside Python source code.

API keys should be loaded through environment variables.

Future Improvements
Hybrid keyword and semantic retrieval
Retrieval reranking
Improved table querying
Better metadata search
Web research integration
Advanced agent planning
Retrieval evaluation metrics
Production authentication
Cloud deployment
Persistent database storage
License

This project is currently intended for educational and development purposes.
