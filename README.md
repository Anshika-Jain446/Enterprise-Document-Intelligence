# 📄 Enterprise Document Intelligence

An agentic Retrieval-Augmented Generation (RAG) platform for enterprise documents. Upload PDFs, Word docs, spreadsheets, and presentations, then ask natural-language questions and get grounded, source-cited answers — with automatic fallback to live web search when your documents don't have the answer.

Built with **Python, Streamlit, PostgreSQL, and Google Gemini**.

---

## ✨ Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, and Markdown
- **Configurable chunking** — six strategies: Character, Recursive, Token, Markdown, Context, and Table
- **Agentic retrieval** — the LLM plans the best retrieval action (document search, table search, or web search) per question, and evaluates whether the evidence is actually sufficient before answering
- **Web search fallback** — when local documents don't have the answer, the assistant automatically searches the web via Gemini's built-in Google Search grounding
- **Evidence-grounded answers** — every response shows the exact source chunks, filenames, pages, and relevance scores it was built from, including full table rendering for tabular evidence
- **Document management** — upload, filter, scope chat to specific documents, and download the original file back
- **Persistent multi-turn conversations** — auto-titled, switchable, and deletable
- **Auth & roles** — hashed passwords, first registered account becomes admin, full admin console (user management, system stats)
- **Multi-user, transactional storage** — every upload is a single atomic PostgreSQL transaction; data is scoped per user

---

## 🏗️ Architecture

**Data preparation:** Upload → format-specific extraction → configurable chunking → atomic save to PostgreSQL (document + chunks + metadata).

**Retrieval & answering:** Question → LLM plans a retrieval action → PostgreSQL full-text search (`tsvector`/`ts_rank`) → LLM evaluates whether the evidence is sufficient → if not, falls back to live web search → LLM generates a grounded, cited answer.

> **Note:** Despite the internal action name `vector_search`, retrieval is PostgreSQL full-text (keyword/relevance) search, not embedding-based semantic search. No vector database is actually used.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| App / UI | Python, Streamlit |
| Database | PostgreSQL |
| Auth | Werkzeug (password hashing) |
| Document parsing | pypdf, python-docx, python-pptx, pandas, openpyxl |
| Chunking | Custom `ChunkingEngine` (langchain-text-splitters, tiktoken) |
| Retrieval | PostgreSQL full-text search |
| LLM | Google Gemini API (`gemini-3.6-flash`, free tier) via `google-genai` |
| Deployment | Railway (containerized app + linked PostgreSQL service) |

---

## 📁 Project Structure

```
.
├── app.py            # Streamlit app: UI, auth, document pipeline, chat, admin
├── model.py           # GeminiModel — LLM client (Gemini API)
├── llm.py              # GeminiLLM — alternate LLM client with deterministic routing
├── config.py          # Environment configuration
├── chunking.py        # ChunkingEngine — six chunking strategies
└── requirements.txt
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full PostgreSQL connection string |
| `GOOGLE_API_KEY` | Yes | — | Free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `LLM_MODEL` | No | `gemini-2.5-flash` | Any Gemini model name |
| `ADMIN_USERNAME` | No | — | Promotes an existing username to admin on startup |
| `CHUNK_SIZE` | No | `1000` | Default chunk size |
| `CHUNK_OVERLAP` | No | `200` | Default chunk overlap |

Alternatively, set `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` individually instead of `DATABASE_URL`.

---

## 🚀 Local Setup

```bash
git clone https://github.com/Anshika-Jain446/Enterprise-Document-Intelligence.git
cd Enterprise-Document-Intelligence
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
DATABASE_URL=postgresql://user:password@host:5432/dbname
GOOGLE_API_KEY=your_free_gemini_api_key
```

Run it:

```bash
streamlit run app.py
```

The first account you register automatically becomes an administrator.

---

## ☁️ Deployment (Railway)

1. Create a new Railway project and add a **PostgreSQL** service.
2. Add your app as a second service in the same project, deployed from this GitHub repo.
3. In the app service's **Variables**, set:
   - `DATABASE_URL` = `${{Postgres.DATABASE_URL}}`
   - `GOOGLE_API_KEY` = your Gemini API key
4. In **Settings → Deploy → Start Command**, set:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```
5. In **Settings → Networking**, make sure the public domain's target port matches the port Streamlit actually binds to (check Deploy Logs for the `Local URL` line if unsure).

---

## 🖱️ Usage

1. **Register / log in** — first account becomes admin.
2. **Documents page** — upload files, choose a chunking method, chunk size, and overlap.
3. **Chat page** — ask questions; optionally scope search to selected documents.
4. Expand **🔎 Retrieved evidence** under any answer to see exactly what it was grounded in.
5. **Admin page** (admins only) — manage users and view system-wide stats.

---

## 📝 Notes

- `EMBEDDING_MODEL` and `VECTOR_DB_PATH` config values exist for forward compatibility but are currently unused — retrieval is full-text search, not embeddings.
- If the Gemini API key isn't configured, the app degrades gracefully: it returns raw matching document evidence instead of an AI-generated answer, rather than failing outright.
