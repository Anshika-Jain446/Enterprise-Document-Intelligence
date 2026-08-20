# app.py

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from werkzeug.security import generate_password_hash, check_password_hash

from chunking import ChunkingEngine
from model import GeminiModel


# ============================================================
# OPTIONAL POSTGRESQL DRIVER
# ============================================================

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    Json = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("POSTGRESQL_URL", "")
)

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE", "enterprise_rag")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")


# ============================================================
# POSTGRESQL DATABASE
# ============================================================

class PostgreSQLStore:

    def __init__(self):
        self.connection = None

    def connect(self):

        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is not installed. "
                "Install it with: pip install psycopg2-binary"
            )

        if self.connection is not None:
            try:
                if self.connection.closed == 0:
                    return self.connection
            except Exception:
                pass

        if DATABASE_URL:
            self.connection = psycopg2.connect(DATABASE_URL)
        else:
            self.connection = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD,
            )

        self.connection.autocommit = False

        return self.connection

    def initialize(self):

        conn = self.connect()
        cursor = conn.cursor()

        try:

            # ====================================================
            # USERS
            # ====================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # ----------------------------------------------------
            # FIX EXISTING USERS TABLE
            # ----------------------------------------------------

            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conrelid = 'users'::regclass
                    AND contype = 'p'
                    AND conkey = ARRAY[
                        (
                            SELECT attnum
                            FROM pg_attribute
                            WHERE attrelid = 'users'::regclass
                            AND attname = 'id'
                        )
                    ]::smallint[]
                )
                """
            )

            users_id_is_primary = cursor.fetchone()[0]

            if not users_id_is_primary:

                cursor.execute(
                    """
                    SELECT id
                    FROM users
                    GROUP BY id
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                )

                duplicate_id = cursor.fetchone()

                if duplicate_id:
                    raise RuntimeError(
                        "Existing users table contains duplicate IDs. "
                        "Please repair the PostgreSQL users table."
                    )

                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid = 'users'::regclass
                        AND contype = 'u'
                        AND conkey = ARRAY[
                            (
                                SELECT attnum
                                FROM pg_attribute
                                WHERE attrelid = 'users'::regclass
                                AND attname = 'id'
                            )
                        ]::smallint[]
                    )
                    """
                )

                users_id_is_unique = cursor.fetchone()[0]

                if not users_id_is_unique:
                    cursor.execute(
                        """
                        ALTER TABLE users
                        ADD CONSTRAINT users_id_unique
                        UNIQUE (id)
                        """
                    )

            # ====================================================
            # DOCUMENTS
            # ====================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,

                    user_id INTEGER NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,

                    filename TEXT NOT NULL,
                    file_type TEXT,
                    file_size BIGINT,
                    file_data BYTEA,

                    chunking_method TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,

                    metadata JSONB DEFAULT '{}'::jsonb,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(user_id, filename)
                );
                """
            )

            # ====================================================
            # DOCUMENT CHUNKS
            # ====================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,

                    document_id INTEGER NOT NULL
                        REFERENCES documents(id)
                        ON DELETE CASCADE,

                    chunk_id INTEGER NOT NULL,
                    chunk_type TEXT,
                    content TEXT,

                    page INTEGER,
                    tokens INTEGER,
                    characters INTEGER,

                    metadata JSONB DEFAULT '{}'::jsonb,
                    chunk_data JSONB DEFAULT '{}'::jsonb,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(document_id, chunk_id)
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_document_chunks_document
                ON document_chunks(document_id);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_document_chunks_content
                ON document_chunks
                USING gin(
                    to_tsvector(
                        'english',
                        COALESCE(content, '')
                    )
                );
                """
            )

            # ====================================================
            # CONVERSATIONS
            # ====================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,

                    user_id INTEGER NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,

                    title TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # ====================================================
            # MESSAGES
            # ====================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,

                    conversation_id INTEGER NOT NULL
                        REFERENCES conversations(id)
                        ON DELETE CASCADE,

                    role TEXT NOT NULL,
                    content TEXT NOT NULL,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()

    # ========================================================
    # USER
    # ========================================================

    def create_user(self, username, password):

        conn = self.connect()
        cursor = conn.cursor()

        password_hash = generate_password_hash(password)

        try:

            cursor.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash
                )
                VALUES (%s, %s)
                RETURNING id, username
                """,
                (
                    username,
                    password_hash,
                ),
            )

            result = cursor.fetchone()

            conn.commit()

            return result

        except Exception:

            conn.rollback()

            return None

        finally:
            cursor.close()

    # ========================================================
    # LOGIN
    # ========================================================

    def authenticate(self, username, password):

        conn = self.connect()

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                id,
                username,
                password_hash
            FROM users
            WHERE username = %s
            """,
            (username,),
        )

        user = cursor.fetchone()

        cursor.close()

        if not user:
            return None

        if not check_password_hash(
            user["password_hash"],
            password,
        ):
            return None

        return dict(user)

    # ========================================================
    # SAVE DOCUMENT
    # ========================================================

    def save_document(
        self,
        user_id,
        filename,
        file_type,
        file_bytes,
        chunking_method,
        chunk_size,
        chunk_overlap,
        metadata,
    ):

        conn = self.connect()
        cursor = conn.cursor()

        try:

            cursor.execute(
                """
                DELETE FROM documents
                WHERE user_id = %s
                AND filename = %s
                """,
                (
                    user_id,
                    filename,
                ),
            )

            cursor.execute(
                """
                INSERT INTO documents (
                    user_id,
                    filename,
                    file_type,
                    file_size,
                    file_data,
                    chunking_method,
                    chunk_size,
                    chunk_overlap,
                    metadata
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    user_id,
                    filename,
                    file_type,
                    len(file_bytes),
                    psycopg2.Binary(file_bytes),
                    chunking_method,
                    chunk_size,
                    chunk_overlap,
                    Json(metadata),
                ),
            )

            document_id = cursor.fetchone()[0]

            conn.commit()

            return document_id

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()

    # ========================================================
    # SAVE CHUNKS
    # ========================================================

    def save_chunks(self, document_id, chunks):

        conn = self.connect()
        cursor = conn.cursor()

        try:

            for chunk in chunks:

                metadata = chunk.get(
                    "metadata",
                    {}
                )

                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id,
                        chunk_id,
                        chunk_type,
                        content,
                        page,
                        tokens,
                        characters,
                        metadata,
                        chunk_data
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (
                        document_id,
                        chunk_id
                    )
                    DO UPDATE SET
                        chunk_type = EXCLUDED.chunk_type,
                        content = EXCLUDED.content,
                        page = EXCLUDED.page,
                        tokens = EXCLUDED.tokens,
                        characters = EXCLUDED.characters,
                        metadata = EXCLUDED.metadata,
                        chunk_data = EXCLUDED.chunk_data
                    """,
                    (
                        document_id,
                        chunk.get("chunk_id"),
                        chunk.get("chunk_type"),
                        chunk.get("content", ""),
                        chunk.get("page"),
                        chunk.get("tokens"),
                        chunk.get(
                            "characters",
                            len(
                                str(
                                    chunk.get(
                                        "content",
                                        "",
                                    )
                                )
                            ),
                        ),
                        Json(metadata),
                        Json(chunk),
                    ),
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()

    # ========================================================
    # DOCUMENT LIST
    # ========================================================

    def get_documents(self, user_id):

        conn = self.connect()

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                id,
                filename,
                file_type,
                file_size,
                chunking_method,
                chunk_size,
                chunk_overlap,
                created_at
            FROM documents
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )

        rows = cursor.fetchall()
        cursor.close()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # SEARCH DOCUMENT CHUNKS
    # ========================================================

    def search_chunks(
        self,
        user_id,
        query,
        selected_document_ids=None,
        chunk_types=None,
        top_k=5,
    ):

        conn = self.connect()

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        conditions = [
            "d.user_id = %s"
        ]

        params = [
            user_id
        ]

        if selected_document_ids:

            conditions.append(
                "d.id = ANY(%s)"
            )

            params.append(
                selected_document_ids
            )

        if chunk_types:

            conditions.append(
                "c.chunk_type = ANY(%s)"
            )

            params.append(
                chunk_types
            )

        search_query = f"""
            SELECT
                c.id,
                c.document_id,
                c.chunk_id,
                c.chunk_type,
                c.content,
                c.page,
                c.tokens,
                c.characters,
                c.metadata,
                c.chunk_data,
                d.filename,
                d.chunking_method,

                ts_rank(
                    to_tsvector(
                        'english',
                        COALESCE(c.content, '')
                    ),
                    plainto_tsquery(
                        'english',
                        %s
                    )
                ) AS similarity_score

            FROM document_chunks c

            JOIN documents d
                ON d.id = c.document_id

            WHERE
                {" AND ".join(conditions)}

            AND (
                to_tsvector(
                    'english',
                    COALESCE(c.content, '')
                )
                @@ plainto_tsquery(
                    'english',
                    %s
                )

                OR

                LOWER(c.content)
                LIKE LOWER(%s)
            )

            ORDER BY similarity_score DESC

            LIMIT %s
        """

        params.extend(
            [
                query,
                query,
                f"%{query}%",
                top_k,
            ]
        )

        cursor.execute(
            search_query,
            params,
        )

        rows = cursor.fetchall()
        cursor.close()

        results = []

        for row in rows:

            result = dict(row)

            result["source"] = result.get(
                "filename"
            )

            result["search_type"] = (
                "postgresql_chunk_search"
            )

            result["source_type"] = (
                "table"
                if result.get("chunk_type") == "Table"
                else "document"
            )

            results.append(result)

        return results

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    def create_conversation(
        self,
        user_id,
        title="New Conversation",
    ):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (
                user_id,
                title
            )
            VALUES (%s, %s)
            RETURNING id
            """,
            (
                user_id,
                title,
            ),
        )

        conversation_id = cursor.fetchone()[0]

        conn.commit()
        cursor.close()

        return conversation_id

    # ========================================================
    # SAVE MESSAGE
    # ========================================================

    def save_message(
        self,
        conversation_id,
        role,
        content,
    ):

        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content
            )
            VALUES (%s, %s, %s)
            """,
            (
                conversation_id,
                role,
                content,
            ),
        )

        conn.commit()
        cursor.close()

    # ========================================================
    # GET MESSAGES
    # ========================================================

    def get_messages(self, conversation_id):

        conn = self.connect()

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                role,
                content,
                created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at
            """,
            (conversation_id,),
        )

        rows = cursor.fetchall()
        cursor.close()

        return [
            dict(row)
            for row in rows
        ]


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_file(uploaded_file):

    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()

    file_bytes = uploaded_file.getvalue()

    text = ""

    pages = []
    tables = []
    images = []
    visuals = []

    metadata = {
        "uploaded_file_name": filename,
        "filename": filename,
        "file_type": suffix,
        "file_size": len(file_bytes),
    }

    # ========================================================
    # TXT / MD / CSV
    # ========================================================

    if suffix in {
        ".txt",
        ".md",
        ".csv",
    }:

        text = file_bytes.decode(
            "utf-8",
            errors="ignore",
        )

        if suffix == ".csv":

            try:

                dataframe = pd.read_csv(
                    io.BytesIO(file_bytes)
                )

                rows = (
                    dataframe
                    .fillna("")
                    .to_dict(
                        orient="records"
                    )
                )

                tables.append(
                    {
                        "table": rows,
                        "page": 1,
                        "source": filename,
                    }
                )

            except Exception:
                pass

    # ========================================================
    # PDF
    # ========================================================

    elif suffix == ".pdf":

        try:

            from pypdf import PdfReader

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            page_texts = []

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):

                page_text = (
                    page.extract_text()
                    or ""
                )

                page_texts.append(
                    page_text
                )

                pages.append(
                    {
                        "page": page_number,
                        "text": page_text,
                    }
                )

            text = "\n\n".join(
                page_texts
            )

            metadata["page_count"] = len(
                reader.pages
            )

        except Exception as exc:

            st.warning(
                f"PDF extraction failed: {exc}"
            )

    # ========================================================
    # DOCX
    # ========================================================

    elif suffix == ".docx":

        try:

            from docx import Document

            document = Document(
                io.BytesIO(file_bytes)
            )

            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            text = "\n\n".join(
                paragraphs
            )

            for table in document.tables:

                rows = []

                for row in table.rows:

                    values = [
                        cell.text
                        for cell in row.cells
                    ]

                    rows.append(values)

                tables.append(
                    {
                        "table": rows,
                        "page": 1,
                        "source": filename,
                    }
                )

        except Exception as exc:

            st.warning(
                f"DOCX extraction failed: {exc}"
            )

    # ========================================================
    # XLSX
    # ========================================================

    elif suffix == ".xlsx":

        try:

            workbook = pd.ExcelFile(
                io.BytesIO(file_bytes)
            )

            for sheet_name in workbook.sheet_names:

                dataframe = pd.read_excel(
                    workbook,
                    sheet_name=sheet_name,
                )

                rows = (
                    dataframe
                    .fillna("")
                    .to_dict(
                        orient="records"
                    )
                )

                tables.append(
                    {
                        "table": rows,
                        "sheet": sheet_name,
                        "page": sheet_name,
                        "source": filename,
                    }
                )

                text += (
                    f"\n\n"
                    f"Sheet: {sheet_name}\n"
                    f"{dataframe.to_string(index=False)}"
                )

        except Exception as exc:

            st.warning(
                f"XLSX extraction failed: {exc}"
            )

    return {
        "text": text,
        "metadata": metadata,
        "tables": tables,
        "images": images,
        "visuals": visuals,
        "pages": pages,
        "source": filename,
        "filename": filename,
        "file_name": filename,
    }


# ============================================================
# CHUNKING
# ============================================================

CHUNKING_METHODS = {
    "Recursive": "recursive_chunking",
    "Character": "character_chunking",
    "Token": "token_chunking",
    "Markdown": "markdown_chunking",
    "Contextual": "contextual_chunking",
    "Table": "table_chunking",
    "Image": "image_chunking",
    "Visual": "visual_chunking",
    "Multimodal": "multimodal_chunking",
}


def create_chunks(
    extracted,
    method,
    chunk_size,
    chunk_overlap,
):

    engine = ChunkingEngine(
        text=extracted.get(
            "text",
            "",
        ),
        metadata=extracted.get(
            "metadata",
            {},
        ),
        tables=extracted.get(
            "tables",
            [],
        ),
        images=extracted.get(
            "images",
            [],
        ),
        visuals=extracted.get(
            "visuals",
            [],
        ),
        pages=extracted.get(
            "pages",
            [],
        ),
        source=extracted.get(
            "source",
            "Unknown",
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    method_name = CHUNKING_METHODS[method]

    chunk_function = getattr(
        engine,
        method_name,
    )

    chunks = chunk_function()

    return chunks, engine


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {
        "authenticated": False,
        "user": None,
        "db": None,
        "llm": None,
        "conversation_id": None,
        "messages": [],
        "documents": [],
        "last_chunks": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_state()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

if st.session_state.db is None:

    try:

        st.session_state.db = PostgreSQLStore()

        st.session_state.db.initialize()

    except Exception as exc:

        st.error(
            "PostgreSQL connection failed."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# LOGIN / REGISTER
# ============================================================

if not st.session_state.authenticated:

    st.title(
        "📄 Enterprise Document Intelligence"
    )

    st.caption(
        "PostgreSQL-backed Agentic RAG"
    )

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register",
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            use_container_width=True,
        ):

            if not username or not password:

                st.warning(
                    "Enter username and password."
                )

            else:

                user = (
                    st.session_state.db.authenticate(
                        username,
                        password,
                    )
                )

                if user:

                    st.session_state.authenticated = True
                    st.session_state.user = user

                    st.session_state.conversation_id = (
                        st.session_state.db.create_conversation(
                            user["id"],
                            "Enterprise RAG",
                        )
                    )

                    st.session_state.messages = []

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

    # ========================================================
    # REGISTER
    # ========================================================

    with register_tab:

        new_username = st.text_input(
            "Username",
            key="register_username",
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password",
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="register_confirm",
        )

        if st.button(
            "Create account",
            use_container_width=True,
        ):

            if not new_username or not new_password:

                st.warning(
                    "Username and password are required."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                user = (
                    st.session_state.db.create_user(
                        new_username,
                        new_password,
                    )
                )

                if user:

                    st.success(
                        "Account created. You can now log in."
                    )

                else:

                    st.error(
                        "Username already exists or registration failed."
                    )

    st.stop()


# ============================================================
# LOAD GEMINI MODEL
# ============================================================

if st.session_state.llm is None:

    try:

        st.session_state.llm = GeminiModel()

    except Exception as exc:

        st.error(
            f"LLM initialization failed: {exc}"
        )

        st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 Enterprise Document Intelligence"
)

st.caption(
    "PostgreSQL • Agentic RAG • Multimodal Chunking • "
    "Document Search • Table Search • Web Search"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👤 Account")

    st.write(
        f"Logged in as **{st.session_state.user['username']}**"
    )

    if st.button(
        "Logout",
        use_container_width=True,
    ):

        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.conversation_id = None

        st.rerun()

    st.divider()

    # ========================================================
    # CHUNKING CONFIGURATION
    # ========================================================

    st.header("🧩 Chunking")

    chunking_method = st.selectbox(
        "Chunking method",
        options=list(CHUNKING_METHODS.keys()),
        index=list(
            CHUNKING_METHODS.keys()
        ).index("Multimodal"),
    )

    chunk_size = st.number_input(
        "Chunk size",
        min_value=100,
        max_value=20000,
        value=1000,
        step=100,
    )

    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=5000,
        value=200,
        step=50,
    )

    if chunk_overlap >= chunk_size:

        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

    st.caption(
        "Selected chunking configuration is stored "
        "with the document in PostgreSQL."
    )

    st.divider()

    # ========================================================
    # DOCUMENT UPLOAD
    # ========================================================

    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "txt",
            "md",
            "csv",
            "docx",
            "xlsx",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        if chunk_overlap < chunk_size:

            if st.button(
                "⬆️ Process & Store Documents",
                use_container_width=True,
            ):

                progress = st.progress(0)

                total = len(uploaded_files)

                for index, uploaded_file in enumerate(
                    uploaded_files,
                    start=1,
                ):

                    try:

                        extracted = extract_file(
                            uploaded_file
                        )

                        chunks, engine = create_chunks(
                            extracted,
                            chunking_method,
                            chunk_size,
                            chunk_overlap,
                        )

                        document_id = (
                            st.session_state.db.save_document(
                                user_id=st.session_state.user[
                                    "id"
                                ],
                                filename=uploaded_file.name,
                                file_type=Path(
                                    uploaded_file.name
                                ).suffix.lower(),
                                file_bytes=uploaded_file.getvalue(),
                                chunking_method=chunking_method,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                metadata=extracted.get(
                                    "metadata",
                                    {},
                                ),
                            )
                        )

                        st.session_state.db.save_chunks(
                            document_id,
                            chunks,
                        )

                        progress.progress(
                            index / total
                        )

                        st.success(
                            f"{uploaded_file.name}: "
                            f"{len(chunks)} chunks stored."
                        )

                    except Exception as exc:

                        st.error(
                            f"Failed to process "
                            f"{uploaded_file.name}: {exc}"
                        )

                st.session_state.documents = (
                    st.session_state.db.get_documents(
                        st.session_state.user["id"]
                    )
                )

    # ========================================================
    # EXISTING DOCUMENTS
    # ========================================================

    st.divider()

    documents = (
        st.session_state.db.get_documents(
            st.session_state.user["id"]
        )
    )

    st.session_state.documents = documents

    if documents:

        st.success(
            f"{len(documents)} document(s) in PostgreSQL."
        )

        document_options = {
            document["filename"]: document["id"]
            for document in documents
        }

        selected_document_names = st.multiselect(
            "Search only selected documents",
            options=list(
                document_options.keys()
            ),
            default=list(
                document_options.keys()
            ),
        )

        selected_document_ids = [
            document_options[name]
            for name in selected_document_names
        ]

    else:

        selected_document_ids = []

        st.info(
            "No documents uploaded yet."
        )

    st.divider()

    # ========================================================
    # RETRIEVAL FILTERS
    # ========================================================

    st.header("🔎 Retrieval filters")

    selected_chunk_types = st.multiselect(
        "Chunk types",
        options=[
            "Recursive",
            "Character",
            "Token",
            "Markdown",
            "Context",
            "Table",
            "Image",
            "Visual",
            "Multimodal",
        ],
        default=[],
        help="Leave empty to search every stored chunk.",
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        if st.session_state.conversation_id:

            st.session_state.conversation_id = (
                st.session_state.db.create_conversation(
                    st.session_state.user["id"],
                    "New Conversation",
                )
            )

        st.rerun()


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(query):

    return (
        st.session_state.db.search_chunks(
            user_id=st.session_state.user["id"],
            query=query,
            selected_document_ids=selected_document_ids,
            chunk_types=selected_chunk_types or None,
            top_k=top_k,
        )
    )


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query):

    method = getattr(
        st.session_state.llm,
        "web_search",
        None,
    )

    if not callable(method):

        return []

    try:

        results = method(
            query=query,
            max_results=top_k,
        )

        if not isinstance(results, list):

            return []

        normalized = []

        for result in results:

            if not isinstance(result, dict):

                continue

            item = dict(result)

            item.setdefault(
                "source_type",
                "web",
            )

            item.setdefault(
                "search_type",
                "web_search",
            )

            item.setdefault(
                "content",
                item.get(
                    "text",
                    item.get(
                        "snippet",
                        "",
                    ),
                ),
            )

            normalized.append(item)

        return normalized

    except Exception as exc:

        st.warning(
            f"Web search failed: {exc}"
        )

        return []


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    chunks,
):

    generator = getattr(
        st.session_state.llm,
        "generate_answer",
        None,
    )

    if not callable(generator):

        if chunks:

            return chunks[0].get(
                "content",
                "No answer available.",
            )

        return (
            "I could not find sufficient evidence."
        )

    try:

        return generator(
            query=query,
            chunks=chunks,
            source_type=(
                "web"
                if chunks
                and all(
                    item.get("source_type") == "web"
                    for item in chunks
                )
                else "document"
            ),
        )

    except Exception as exc:

        st.error(
            f"Answer generation failed: {exc}"
        )

        return (
            "I could not generate an answer "
            "from the retrieved evidence."
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get("role")

    content = message.get(
        "content",
        "",
    )

    if role not in {
        "user",
        "assistant",
    }:

        continue

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    if st.session_state.conversation_id:

        st.session_state.db.save_message(
            st.session_state.conversation_id,
            "user",
            query,
        )

    with st.chat_message("user"):

        st.markdown(query)

    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message("assistant"):

        # ----------------------------------------------------
        # DOCUMENT SEARCH FIRST
        # ----------------------------------------------------

        with st.spinner(
            "Searching PostgreSQL document chunks..."
        ):

            document_results = search_documents(
                query
            )

        # ----------------------------------------------------
        # WEB SEARCH FALLBACK
        # ----------------------------------------------------

        if document_results:

            results = document_results

            search_mode = "Documents"

        else:

            with st.spinner(
                "No document evidence found. Searching web..."
            ):

                web_results = web_search(
                    query
                )

            results = web_results

            search_mode = "Web"

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        if results:

            with st.spinner(
                f"Generating answer from {search_mode}..."
            ):

                answer = generate_answer(
                    query,
                    results,
                )

        else:

            answer = (
                "I could not find sufficient "
                "evidence in the uploaded documents "
                "or available web search."
            )

        st.markdown(answer)

        # ====================================================
        # METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Retrieved results",
                len(results),
            )

        with col2:

            st.metric(
                "Search",
                search_mode,
            )

        with col3:

            st.metric(
                "Storage",
                "PostgreSQL",
            )

        # ====================================================
        # SOURCES
        # ====================================================

        if results:

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for index, source in enumerate(
                    results,
                    start=1,
                ):

                    source_name = (
                        source.get("filename")
                        or source.get("source")
                        or source.get("title")
                        or source.get("url")
                        or "Unknown"
                    )

                    chunk_type = source.get(
                        "chunk_type",
                        "Web",
                    )

                    score = source.get(
                        "similarity_score",
                        "",
                    )

                    page = source.get(
                        "page",
                        "",
                    )

                    st.markdown(
                        f"### {index}. {source_name}"
                    )

                    st.caption(
                        f"Source type: "
                        f"{source.get('source_type', 'unknown')}"
                    )

                    st.caption(
                        f"Chunk type: {chunk_type}"
                    )

                    if page:

                        st.caption(
                            f"Page/Sheet: {page}"
                        )

                    if score != "":

                        st.caption(
                            f"Relevance: {score}"
                        )

                    content = (
                        source.get("content")
                        or source.get("text")
                        or source.get("snippet")
                        or ""
                    )

                    if content:

                        st.text(
                            str(content)[:3000]
                        )

                    result_url = source.get("url")

                    if result_url:

                        st.link_button(
                            "Open web source",
                            result_url,
                        )

                    st.divider()

    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    if st.session_state.conversation_id:

        st.session_state.db.save_message(
            st.session_state.conversation_id,
            "assistant",
            answer,
        )

    st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        ## 👋 Welcome

        Your documents, original files, users,
        conversations, and chunks are stored in
        **PostgreSQL**.

        ### Current pipeline

        `Login`
        →
        `Upload`
        →
        `Extract`
        →
        `Select Chunking`
        →
        `Chunk`
        →
        `Store Chunks in PostgreSQL`
        →
        `Retrieve`
        →
        `Web Search if needed`
        →
        `Gemini`
        →
        `Answer`

        ### Supported chunking

        - Recursive
        - Character
        - Token
        - Markdown
        - Contextual
        - Table
        - Image
        - Visual
        - Multimodal

        ### Search

        1. PostgreSQL document search is attempted first.
        2. If no document evidence is found,
           web search is used.
        3. Gemini generates the final answer.

        Upload a document from the sidebar to begin.
        """
    )