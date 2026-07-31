# 📄 Enterprise Document Intelligence System

An AI-powered Enterprise Document Intelligence System built using **Python**, **Streamlit**, **Sentence Transformers**, and **FAISS**. This project preprocesses enterprise documents for Retrieval-Augmented Generation (RAG) by extracting document content, chunking text, generating embeddings, storing them in a vector database, and performing semantic search.

---

## 🚀 Features

- Upload one or multiple PDF documents
- Extract:
  - Text
  - Metadata
  - Tables
- Multiple chunking techniques:
  - Character Chunking
  - Recursive Chunking
  - Token Chunking
  - Markdown Chunking
  - Contextual Chunking
  - Table Chunking
- Generate sentence embeddings using Sentence Transformers
- Store embeddings in a FAISS Vector Database
- Perform Semantic Search over uploaded documents
- Interactive Streamlit User Interface

---

## 🏗️ Project Workflow

1. Upload PDF documents.
2. Extract document text, metadata, and tables.
3. Perform document chunking.
4. Generate embeddings for each chunk.
5. Store embeddings in a FAISS Vector Database.
6. Enter a user query.
7. Retrieve the most relevant document chunks using semantic similarity.

---

## 📂 Project Structure

```
Enterprise-Document-Intelligence/
│
├── app.py
├── extractor.py
├── chunking.py
├── vector_db.py
├── config.py
├── requirements.txt
├── README.md
│
├── outputs/
│   └── vector_db/
│
└── data/
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- PyMuPDF
- Camelot / pdfplumber (Table Extraction)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Enterprise-Document-Intelligence.git
```

Move into the project directory:

```bash
cd Enterprise-Document-Intelligence
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Start the Streamlit application:

```bash
streamlit run app.py
```

---

## 📖 Usage

1. Launch the application.
2. Upload one or more PDF documents.
3. View extracted metadata and tables.
4. Select a chunking technique.
5. Create the Vector Database.
6. Ask questions related to the uploaded documents.
7. View the retrieved relevant chunks.

---

## 🔍 Example Queries

- What is the purpose of this document?
- Summarize the uploaded document.
- What are the key findings?
- What tables are available in the document?
- Who is the author of the document?
- What are the financial figures mentioned?
- Explain the methodology section.
- What conclusions are provided?
- Find information about revenue.
- Which page discusses project objectives?

---

## 📌 Current Progress

✅ PDF Upload

✅ Text Extraction

✅ Metadata Extraction

✅ Table Extraction

✅ Multiple Chunking Techniques

✅ Embedding Generation

✅ FAISS Vector Database

✅ Semantic Search

### Upcoming Features

- Retrieval-Augmented Generation (RAG)
- LLM Integration
- Conversational AI Chatbot
- Source Citation
- Hybrid Search
- Document Summarization

---

## 👩‍💻 Author

**Anshika Jain**

Enterprise AI | Retrieval-Augmented Generation (RAG) | NLP | Document Intelligence

---

## 📜 License

This project is developed for educational and research purposes.
