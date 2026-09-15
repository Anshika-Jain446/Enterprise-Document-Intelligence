import io
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse, quote

import pandas as pd
import streamlit as st

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from chunking import ChunkingEngine
from model import GeminiModel
from auto_chunking import get_automatic_chunking_method
from vector_db import VectorDatabase
from multi_document_retriever import MultiDocumentRetriever


# ============================================================
# CONFIGURATION
# ============================================================

try:
    import config
except ImportError:
    config = None


def config_value(name, default=None):
    value = os.getenv(name)

    if value is not None:
        return value

    if config is not None:
        return getattr(config, name, default)

    return default


GOOGLE_API_KEY = str(
    config_value("GOOGLE_API_KEY", "")
    or config_value("GEMINI_API_KEY", "")
    or ""
).strip()

LLM_MODEL = str(
    config_value(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )
    or "gemini-3.6-flash"
).strip()

POSTGRES_HOST = str(
    config_value("POSTGRES_HOST", "localhost")
    or "localhost"
).strip()

try:
    POSTGRES_PORT = int(
        config_value("POSTGRES_PORT", 5432) or 5432
    )
except Exception:
    POSTGRES_PORT = 5432

POSTGRES_DB = str(
    config_value("POSTGRES_DB", "") or ""
).strip()

POSTGRES_USER = str(
    config_value("POSTGRES_USER", "") or ""
).strip()

POSTGRES_PASSWORD = str(
    config_value("POSTGRES_PASSWORD", "") or ""
)

try:
    CHUNK_SIZE = int(
        config_value("CHUNK_SIZE", 1000) or 1000
    )
except Exception:
    CHUNK_SIZE = 1000

try:
    CHUNK_OVERLAP = int(
        config_value("CHUNK_OVERLAP", 100) or 100
    )
except Exception:
    CHUNK_OVERLAP = 100


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


SUPPORTED_FILES_CONFIG = config_value(
    "SUPPORTED_FILES",
    [
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".csv",
        ".txt",
        ".md",
        ".markdown",
        *sorted(IMAGE_EXTENSIONS),
    ],
)

if isinstance(SUPPORTED_FILES_CONFIG, str):
    SUPPORTED_FILES = [
        item.strip()
        for item in SUPPORTED_FILES_CONFIG.split(",")
        if item.strip()
    ]
else:
    SUPPORTED_FILES = list(
        SUPPORTED_FILES_CONFIG or []
    )


SUPPORTED_EXTENSIONS = {
    (
        str(item).lower()
        if str(item).startswith(".")
        else f".{str(item).lower()}"
    )
    for item in SUPPORTED_FILES
}

# Images are always accepted; they are read with OCR.
SUPPORTED_EXTENSIONS |= IMAGE_EXTENSIONS


DATABASE_URL = str(
    os.getenv(
        "DATABASE_URL",
        os.getenv("POSTGRESQL_URL", ""),
    )
    or ""
).strip()

ADMIN_USERNAME = str(
    os.getenv("ADMIN_USERNAME", "") or ""
).strip()

DEFAULT_CHUNK_SIZE = CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP
DEFAULT_TOP_K = 5


# ============================================================
# OPTIONAL POSTGRESQL IMPORTS
# ============================================================

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    Json = None


try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None


# ============================================================
# DOCUMENT READER IMPORTS
# ============================================================

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


try:
    from pptx import Presentation
except ImportError:
    Presentation = None


try:
    from PIL import Image
except ImportError:
    Image = None


try:
    import pytesseract
except ImportError:
    pytesseract = None


try:
    from ocr import OCRProcessor
except ImportError:
    OCRProcessor = None


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE STORE
# ============================================================

class PostgreSQLStore:

    def __init__(self):
        self.connection = None

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self):

        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is not installed. "
                "Install psycopg2-binary."
            )

        # Reuse healthy connection.
        if self.connection is not None:

            try:
                if self.connection.closed == 0:
                    return self.connection
            except Exception:
                pass

            self.reset_connection()

        try:

            if DATABASE_URL:

                database_url = DATABASE_URL

                # ------------------------------------------------
                # Strip accidental surrounding quotes from the
                # secret value (e.g. pasted as "postgresql://...").
                # ------------------------------------------------
                if (
                    len(database_url) >= 2
                    and database_url[0] == database_url[-1]
                    and database_url[0] in {"'", '"'}
                ):
                    database_url = database_url[1:-1].strip()

                # ------------------------------------------------
                # Parse the URL so passwords containing special
                # characters do not break the PostgreSQL
                # connection string.
                # ------------------------------------------------
                parsed = urlparse(database_url)

                if not parsed.hostname:
                    raise RuntimeError(
                        "Invalid DATABASE_URL: PostgreSQL hostname is missing."
                    )

                username = parsed.username

                if not username:
                    raise RuntimeError(
                        "Invalid DATABASE_URL: PostgreSQL username is missing."
                    )

                password = parsed.password

                if password is None:
                    raise RuntimeError(
                        "Invalid DATABASE_URL: PostgreSQL password is missing."
                    )

                hostname = parsed.hostname
                port = parsed.port or 5432
                database = parsed.path.lstrip("/")

                if not database:
                    database = "postgres"

                # Rebuild the URL with correctly URL-encoded
                # credentials.
                safe_database_url = urlunparse(
                    (
                        "postgresql",
                        (
                            f"{quote(username, safe='')}:"
                            f"{quote(password, safe='')}@"
                            f"{hostname}:{port}"
                        ),
                        f"/{quote(database, safe='')}",
                        "",
                        parsed.query,
                        "",
                    )
                )

                try:

                    self.connection = psycopg2.connect(
                        safe_database_url,
                        connect_timeout=15,
                        sslmode="require",
                    )

                except psycopg2.OperationalError as exc:

                    error_text = str(exc).lower()

                    if "password authentication failed" in error_text:

                        raise RuntimeError(
                            "Supabase rejected the PostgreSQL password.\n\n"
                            "Check DATABASE_URL in Streamlit Secrets and "
                            "make sure it contains the CURRENT Supabase "
                            "database password.\n\n"
                            "Do not use your Supabase dashboard login "
                            "password. Use the PostgreSQL database "
                            "password."
                        ) from exc

                    raise RuntimeError(
                        f"PostgreSQL connection failed: {exc}"
                    ) from exc

            else:

                if not POSTGRES_DB:
                    raise RuntimeError(
                        "PostgreSQL is not configured. "
                        "Set DATABASE_URL or POSTGRES_DB."
                    )

                self.connection = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    connect_timeout=15,
                )

            self.connection.autocommit = False

            if register_vector is not None:

                try:
                    register_vector(self.connection)
                except Exception:
                    pass

            cursor = self.connection.cursor()

            try:

                cursor.execute(
                    "SET search_path TO public"
                )

                self.connection.commit()

            except Exception:

                self.connection.rollback()
                raise

            finally:
                cursor.close()

            return self.connection

        except Exception:

            self.reset_connection()
            raise

    # ========================================================
    # RESET CONNECTION
    # ========================================================

    def reset_connection(self):

        try:

            if self.connection is not None:
                self.connection.close()

        except Exception:
            pass

        self.connection = None

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def initialize(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            # ------------------------------------------------
            # USERS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(50) NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ------------------------------------------------
            # DOCUMENTS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT,
                    file_size BIGINT,
                    file_data BYTEA,
                    chunking_method TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT documents_user_fk
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                )
                """
            )

            # ------------------------------------------------
            # CHUNKS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    chunk_type TEXT,
                    content TEXT,
                    page TEXT,
                    tokens INTEGER,
                    characters INTEGER,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    chunk_data JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT chunks_document_fk
                    FOREIGN KEY (document_id)
                    REFERENCES public.documents(id)
                    ON DELETE CASCADE,

                    CONSTRAINT document_chunk_unique
                    UNIQUE(document_id, chunk_id)
                )
                """
            )

            # ------------------------------------------------
            # CONVERSATIONS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.conversations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT DEFAULT 'New Conversation',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT conversations_user_fk
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                )
                """
            )

            # ------------------------------------------------
            # MESSAGES
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT messages_conversation_fk
                    FOREIGN KEY (conversation_id)
                    REFERENCES public.conversations(id)
                    ON DELETE CASCADE
                )
                """
            )

            # ------------------------------------------------
            # MIGRATIONS
            # ------------------------------------------------

            cursor.execute(
                """
                ALTER TABLE public.documents
                ADD COLUMN IF NOT EXISTS metadata
                JSONB DEFAULT '{}'::jsonb
                """
            )

            cursor.execute(
                """
                ALTER TABLE public.documents
                ADD COLUMN IF NOT EXISTS chunking_method TEXT
                """
            )

            cursor.execute(
                """
                ALTER TABLE public.documents
                ADD COLUMN IF NOT EXISTS chunk_size INTEGER
                """
            )

            cursor.execute(
                """
                ALTER TABLE public.documents
                ADD COLUMN IF NOT EXISTS chunk_overlap INTEGER
                """
            )

            cursor.execute(
                """
                ALTER TABLE public.document_chunks
                ADD COLUMN IF NOT EXISTS metadata
                JSONB DEFAULT '{}'::jsonb
                """
            )

            cursor.execute(
                """
                ALTER TABLE public.document_chunks
                ADD COLUMN IF NOT EXISTS chunk_data
                JSONB DEFAULT '{}'::jsonb
                """
            )

            # ------------------------------------------------
            # INDEXES
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_user
                ON public.documents(user_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON public.document_chunks(document_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON public.conversations(user_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON public.messages(conversation_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_content
                ON public.document_chunks
                USING gin(
                    to_tsvector(
                        'english',
                        COALESCE(content, '')
                    )
                )
                """
            )

            # ------------------------------------------------
            # ADMIN SETUP
            # ------------------------------------------------

            cursor.execute(
                """
                SELECT id
                FROM public.users
                WHERE LOWER(COALESCE(role, 'user')) = 'admin'
                LIMIT 1
                """
            )

            admin = cursor.fetchone()

            if not admin and ADMIN_USERNAME:

                cursor.execute(
                    """
                    UPDATE public.users
                    SET role = 'admin'
                    WHERE LOWER(username) = LOWER(%s)
                    """,
                    (ADMIN_USERNAME,),
                )

            # If no configured admin exists, make the oldest
            # existing user an admin.
            cursor.execute(
                """
                SELECT id
                FROM public.users
                WHERE LOWER(COALESCE(role, 'user')) = 'admin'
                LIMIT 1
                """
            )

            admin = cursor.fetchone()

            if not admin:

                cursor.execute(
                    """
                    SELECT id
                    FROM public.users
                    ORDER BY id
                    LIMIT 1
                    """
                )

                first_user = cursor.fetchone()

                if first_user:

                    cursor.execute(
                        """
                        UPDATE public.users
                        SET role = 'admin'
                        WHERE id = %s
                        """,
                        (first_user[0],),
                    )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # USER LOOKUP
    # ========================================================

    def get_user_by_id(self, user_id):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    role
                FROM public.users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            row = cursor.fetchone()

            conn.rollback()

            return dict(row) if row else None

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # CREATE USER
    # ========================================================

    def create_user(self, username, password):

        username = str(username or "").strip()

        if not username:
            return None

        if not password:
            return None

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            # First user becomes admin.
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.users
                """
            )

            user_count = int(
                cursor.fetchone()[0]
            )

            role = (
                "admin"
                if user_count == 0
                else "user"
            )

            password_hash = generate_password_hash(
                password
            )

            cursor.execute(
                """
                INSERT INTO public.users
                (
                    username,
                    password_hash,
                    role
                )
                VALUES (%s, %s, %s)
                RETURNING id, username, role
                """,
                (
                    username,
                    password_hash,
                    role,
                ),
            )

            result = cursor.fetchone()

            conn.commit()

            return result

        except Exception as exc:

            conn.rollback()

            # Duplicate username.
            if psycopg2 is not None and isinstance(
                exc,
                psycopg2.IntegrityError,
            ):
                return None

            return None

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # AUTHENTICATE
    # ========================================================

    def authenticate(self, username, password):

        username = str(username or "").strip()

        if not username or not password:
            return None

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    role
                FROM public.users
                WHERE LOWER(username) = LOWER(%s)
                """,
                (username,),
            )

            user = cursor.fetchone()

            conn.rollback()

            if not user:
                return None

            stored_hash = user.get(
                "password_hash"
            )

            if not stored_hash:
                return None

            try:

                valid = check_password_hash(
                    stored_hash,
                    password,
                )

            except Exception:

                valid = False

            if not valid:
                return None

            return dict(user)

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # GET USERS
    # ========================================================

    def get_users(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    role,
                    created_at
                FROM public.users
                ORDER BY id
                """
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # CHANGE USER ROLE
    # ========================================================

    def set_user_role(self, user_id, role):

        role = str(role or "").lower()

        if role not in {"admin", "user"}:
            raise ValueError(
                "Invalid role."
            )

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT role
                FROM public.users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            current = cursor.fetchone()

            if not current:
                conn.rollback()
                return False

            current_role = str(
                current[0] or "user"
            ).lower()

            if (
                current_role == "admin"
                and role == "user"
            ):

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(
                        COALESCE(role, 'user')
                    ) = 'admin'
                    """
                )

                admin_count = int(
                    cursor.fetchone()[0]
                )

                if admin_count <= 1:
                    conn.rollback()
                    return False

            cursor.execute(
                """
                UPDATE public.users
                SET role = %s
                WHERE id = %s
                """,
                (
                    role,
                    int(user_id),
                ),
            )

            changed = cursor.rowcount > 0

            conn.commit()

            return changed

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # DELETE USER
    # ========================================================

    def delete_user(self, user_id):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT role
                FROM public.users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            user = cursor.fetchone()

            if not user:
                conn.rollback()
                return False

            if str(
                user[0] or "user"
            ).lower() == "admin":

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(
                        COALESCE(role, 'user')
                    ) = 'admin'
                    """
                )

                admin_count = int(
                    cursor.fetchone()[0]
                )

                if admin_count <= 1:
                    conn.rollback()
                    return False

            cursor.execute(
                """
                DELETE FROM public.users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            deleted = cursor.rowcount > 0

            conn.commit()

            return deleted

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # RESET PASSWORD
    # ========================================================

    def reset_user_password(
        self,
        user_id,
        password,
    ):

        if not password:
            raise ValueError(
                "Password cannot be empty."
            )

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            password_hash = generate_password_hash(
                password
            )

            cursor.execute(
                """
                UPDATE public.users
                SET password_hash = %s
                WHERE id = %s
                """,
                (
                    password_hash,
                    int(user_id),
                ),
            )

            if cursor.rowcount == 0:
                raise RuntimeError(
                    "User does not exist."
                )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # SAVE DOCUMENT + CHUNKS ATOMICALLY
    # ========================================================

    def save_document_with_chunks(
        self,
        user_id,
        filename,
        file_type,
        file_bytes,
        chunking_method,
        chunk_size,
        chunk_overlap,
        chunks,
        metadata=None,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            user_id = int(user_id)

            # ------------------------------------------------
            # Verify owner.
            # ------------------------------------------------

            cursor.execute(
                """
                SELECT id
                FROM public.users
                WHERE id = %s
                FOR SHARE
                """,
                (user_id,),
            )

            if not cursor.fetchone():

                raise RuntimeError(
                    "The logged-in user no longer exists."
                )

            # ------------------------------------------------
            # Remove old version.
            # ------------------------------------------------

            cursor.execute(
                """
                DELETE FROM public.documents
                WHERE user_id = %s
                AND filename = %s
                """,
                (
                    user_id,
                    filename,
                ),
            )

            # ------------------------------------------------
            # Insert document.
            # ------------------------------------------------

            cursor.execute(
                """
                INSERT INTO public.documents
                (
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
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
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
                    int(chunk_size),
                    int(chunk_overlap),
                    Json(metadata or {}),
                ),
            )

            row = cursor.fetchone()

            if not row:
                raise RuntimeError(
                    "PostgreSQL did not return a document ID."
                )

            document_id = int(row[0])

            # ------------------------------------------------
            # Insert chunks.
            # ------------------------------------------------

            saved_chunks = 0

            for index, original_chunk in enumerate(
                chunks,
                start=1,
            ):

                if not isinstance(
                    original_chunk,
                    dict,
                ):
                    chunk = {
                        "content": str(
                            original_chunk
                        )
                    }
                else:
                    chunk = dict(
                        original_chunk
                    )

                chunk_id = int(
                    chunk.get(
                        "chunk_id",
                        index,
                    )
                )

                content = str(
                    chunk.get(
                        "content",
                        "",
                    )
                    or ""
                )

                if not content.strip():
                    continue

                page = chunk.get("page")

                if page is not None:
                    page = str(page)

                chunk_type = chunk.get(
                    "chunk_type",
                    "Document",
                )

                chunk_metadata = chunk.get(
                    "metadata",
                    {},
                )

                if not isinstance(
                    chunk_metadata,
                    dict,
                ):
                    chunk_metadata = {}

                tokens = chunk.get("tokens")

                if tokens is None:
                    tokens = max(
                        1,
                        len(
                            content.split()
                        ),
                    )

                characters = chunk.get(
                    "characters"
                )

                if characters is None:
                    characters = len(content)

                cursor.execute(
                    """
                    INSERT INTO public.document_chunks
                    (
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
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    ON CONFLICT
                    (
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
                        chunk_id,
                        chunk_type,
                        content,
                        page,
                        int(tokens),
                        int(characters),
                        Json(chunk_metadata),
                        Json(chunk),
                    ),
                )

                saved_chunks += 1

            if saved_chunks == 0:

                raise RuntimeError(
                    "No usable document chunks were generated."
                )

            # ------------------------------------------------
            # Verify chunks.
            # ------------------------------------------------

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.document_chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )

            chunk_count = int(
                cursor.fetchone()[0]
            )

            if chunk_count != saved_chunks:

                raise RuntimeError(
                    "Chunk verification failed. "
                    f"Expected {saved_chunks}, "
                    f"found {chunk_count}."
                )

            # ------------------------------------------------
            # Commit.
            # ------------------------------------------------

            conn.commit()

            return {
                "document_id": document_id,
                "chunk_count": chunk_count,
            }

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # GET DOCUMENTS
    # ========================================================

    def get_documents(
        self,
        user_id=None,
        is_admin=False,
        search=None,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            conditions = []
            params = []

            if not is_admin:

                if user_id is None:
                    return []

                conditions.append(
                    "d.user_id = %s"
                )

                params.append(
                    int(user_id)
                )

            if search and str(search).strip():

                pattern = (
                    "%"
                    + str(search).strip()
                    + "%"
                )

                conditions.append(
                    """
                    (
                        LOWER(d.filename)
                        LIKE LOWER(%s)
                        OR LOWER(
                            COALESCE(
                                d.file_type,
                                ''
                            )
                        )
                        LIKE LOWER(%s)
                    )
                    """
                )

                params.extend(
                    [
                        pattern,
                        pattern,
                    ]
                )

            where_clause = ""

            if conditions:
                where_clause = (
                    "WHERE "
                    + " AND ".join(conditions)
                )

            query = f"""
                SELECT
                    d.id,
                    d.user_id,
                    d.filename,
                    d.file_type,
                    d.file_size,
                    d.chunking_method,
                    d.chunk_size,
                    d.chunk_overlap,
                    d.metadata,
                    d.created_at,
                    u.username,

                    (
                        SELECT COUNT(*)
                        FROM public.document_chunks c
                        WHERE c.document_id = d.id
                    ) AS chunk_count

                FROM public.documents d

                LEFT JOIN public.users u
                    ON u.id = d.user_id

                {where_clause}

                ORDER BY
                    d.created_at DESC,
                    d.id DESC
            """

            cursor.execute(
                query,
                params,
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [
                dict(row)
                for row in rows
            ]

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # DOCUMENT BY ID
    # ========================================================

    def get_document_by_id(
        self,
        document_id,
        user_id=None,
        is_admin=False,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            if is_admin:

                cursor.execute(
                    """
                    SELECT
                        d.*,
                        u.username
                    FROM public.documents d
                    LEFT JOIN public.users u
                        ON u.id = d.user_id
                    WHERE d.id = %s
                    """,
                    (int(document_id),),
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        d.*,
                        u.username
                    FROM public.documents d
                    LEFT JOIN public.users u
                        ON u.id = d.user_id
                    WHERE d.id = %s
                    AND d.user_id = %s
                    """,
                    (
                        int(document_id),
                        int(user_id),
                    ),
                )

            row = cursor.fetchone()

            conn.rollback()

            return dict(row) if row else None

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # DOCUMENT FILE (DOWNLOAD)
    # ========================================================

    def get_document_file(
        self,
        document_id,
        user_id=None,
        is_admin=False,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            if is_admin:

                cursor.execute(
                    """
                    SELECT
                        filename,
                        file_type,
                        file_data
                    FROM public.documents
                    WHERE id = %s
                    """,
                    (int(document_id),),
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        filename,
                        file_type,
                        file_data
                    FROM public.documents
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        int(document_id),
                        int(user_id),
                    ),
                )

            row = cursor.fetchone()

            conn.rollback()

            if not row:
                return None

            file_data = row.get("file_data")

            if file_data is not None:

                file_data = bytes(file_data)

            return {
                "filename": row.get("filename"),
                "file_type": row.get("file_type"),
                "file_data": file_data,
            }

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id,
        user_id=None,
        is_admin=False,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            if is_admin:

                cursor.execute(
                    """
                    DELETE FROM public.documents
                    WHERE id = %s
                    """,
                    (int(document_id),),
                )

            else:

                cursor.execute(
                    """
                    DELETE FROM public.documents
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        int(document_id),
                        int(user_id),
                    ),
                )

            deleted = cursor.rowcount > 0

            conn.commit()

            return deleted

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # SEARCH CHUNKS
    # ========================================================

    def search_chunks(
        self,
        user_id,
        query,
        selected_document_ids=None,
        chunk_types=None,
        top_k=5,
    ):

        query = str(query or "").strip()

        if not query:
            return []

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            # ------------------------------------------------
            # SECURITY CONDITIONS
            # ------------------------------------------------

            conditions = [
                "d.user_id = %s"
            ]

            condition_params = [
                int(user_id)
            ]

            # ------------------------------------------------
            # DOCUMENT FILTER
            # ------------------------------------------------

            document_filter_applied = False

            if selected_document_ids:

                clean_ids = []

                for value in selected_document_ids:

                    try:
                        clean_ids.append(
                            int(value)
                        )
                    except Exception:
                        continue

                if clean_ids:

                    conditions.append(
                        "d.id = ANY(%s)"
                    )

                    condition_params.append(
                        clean_ids
                    )

                    document_filter_applied = True

            # ------------------------------------------------
            # CHUNK TYPE FILTER
            # ------------------------------------------------

            if chunk_types:

                clean_types = [
                    str(x)
                    for x in chunk_types
                    if str(x).strip()
                ]

                if clean_types:

                    conditions.append(
                        "c.chunk_type = ANY(%s)"
                    )

                    condition_params.append(
                        clean_types
                    )

            where_clause = " AND ".join(
                conditions
            )

            # ------------------------------------------------
            # IMPORTANT FIX:
            #
            # The placeholders in SQL occur in this order:
            #
            # 1. similarity query
            # 2. user_id / filters
            # 3. tsquery
            # 4. LIKE content
            # 5. LIKE filename
            # 6. LIMIT
            #
            # The original app.py put user_id first.
            # ------------------------------------------------

            # Token-level matching makes retrieval work for natural-language
            # questions instead of requiring the complete question to exist
            # verbatim in a chunk or filename.
            import re
            stopwords = {
                "what", "whats", "what's", "is", "the", "a", "an", "of",
                "to", "in", "for", "and", "or", "on", "with", "this",
                "that", "are", "was", "were", "do", "does", "did",
                "how", "why", "where", "who", "which", "can", "could",
                "would", "should", "please", "tell", "me", "about",
                "explain", "describe", "show", "give", "from", "into",
                "your", "my", "document"
            }
            query_terms = []
            for term in re.findall(r"[a-zA-Z0-9_\-]{2,}", query.lower()):
                if term not in stopwords and term not in query_terms:
                    query_terms.append(term)
            if not query_terms:
                query_terms = [query.lower()]
            term_patterns = [f"%{term}%" for term in query_terms]

            def build_sql(with_text_match):

                text_match = """
                    AND
                    (
                        to_tsvector(
                            'english',
                            COALESCE(c.content, '')
                        )
                        @@ plainto_tsquery(
                            'english',
                            %s
                        )

                        OR LOWER(
                            COALESCE(c.content, '')
                        )
                        LIKE LOWER(%s)

                        OR LOWER(
                            COALESCE(d.filename, '')
                        )
                        LIKE LOWER(%s)

                        OR LOWER(COALESCE(c.content, '')) LIKE ANY(%s)
                        OR LOWER(COALESCE(d.filename, '')) LIKE ANY(%s)
                    )
                """

                return f"""
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
                    c.created_at,

                    d.filename,
                    d.file_type,
                    d.chunking_method,

                    (
                        ts_rank(
                            to_tsvector(
                                'english',
                                COALESCE(c.content, '')
                            ),
                            plainto_tsquery(
                                'english',
                                %s
                            )
                        )
                        + CASE WHEN LOWER(COALESCE(c.content, '')) LIKE ANY(%s)
                               THEN 0.10 ELSE 0 END
                        + CASE WHEN LOWER(COALESCE(d.filename, '')) LIKE ANY(%s)
                               THEN 0.25 ELSE 0 END
                    ) AS similarity_score

                FROM public.document_chunks c

                INNER JOIN public.documents d
                    ON d.id = c.document_id

                WHERE
                    {where_clause}
                    {text_match if with_text_match else ""}

                ORDER BY
                    similarity_score DESC,
                    c.document_id,
                    c.chunk_id

                LIMIT %s
            """

            limit = max(
                1,
                int(top_k),
            )

            params = [
                query,
                term_patterns,
                term_patterns,
                *condition_params,
                query,
                f"%{query}%",
                f"%{query}%",
                term_patterns,
                term_patterns,
                limit,
            ]

            cursor.execute(
                build_sql(True),
                params,
            )

            rows = cursor.fetchall()

            # ------------------------------------------------
            # TOKEN-LEVEL LEXICAL FALLBACK
            # ------------------------------------------------
            # A natural-language question such as "What is an encoder?"
            # should match the important term "encoder" even when the
            # complete question is not stored verbatim in a chunk.
            # This fallback runs for ALL user documents, not only scoped
            # documents.
            if not rows:
                import re

                stopwords = {
                    "what", "is", "the", "a", "an", "of", "to",
                    "in", "for", "and", "or", "on", "with", "this",
                    "that", "are", "was", "were", "do", "does",
                    "how", "why", "where", "who", "which", "can",
                    "could", "would", "should", "tell", "me",
                }
                terms = []
                for term in re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower()):
                    if term not in stopwords and term not in terms:
                        terms.append(term)

                if terms:
                    term_predicates = []
                    for term in terms:
                        term_predicates.append(
                            "LOWER(COALESCE(c.content, '')) LIKE LOWER(%s)"
                        )
                        term_predicates.append(
                            "LOWER(COALESCE(d.filename, '')) LIKE LOWER(%s)"
                        )

                    score_parts = []
                    score_params = []
                    for term in terms:
                        pattern = f"%{term}%"
                        score_parts.append(
                            "CASE WHEN LOWER(COALESCE(c.content, '')) LIKE LOWER(%s) THEN 1 ELSE 0 END"
                        )
                        score_params.append(pattern)
                        score_parts.append(
                            "CASE WHEN LOWER(COALESCE(d.filename, '')) LIKE LOWER(%s) THEN 1 ELSE 0 END"
                        )
                        score_params.append(pattern)

                    lexical_sql = f"""
                    SELECT
                        c.id, c.document_id, c.chunk_id, c.chunk_type,
                        c.content, c.page, c.tokens, c.characters,
                        c.metadata, c.chunk_data, c.created_at,
                        d.filename, d.file_type, d.chunking_method,
                        ({' + '.join(score_parts)})::float AS similarity_score
                    FROM public.document_chunks c
                    INNER JOIN public.documents d ON d.id = c.document_id
                    WHERE {where_clause}
                      AND ({' OR '.join(term_predicates)})
                    ORDER BY similarity_score DESC, c.document_id, c.chunk_id
                    LIMIT %s
                    """

                    predicate_params = []
                    for term in terms:
                        pattern = f"%{term}%"
                        predicate_params.extend([pattern, pattern])

                    cursor.execute(
                        lexical_sql,
                        [*condition_params, *score_params, *predicate_params, limit],
                    )
                    rows = cursor.fetchall()

            # ------------------------------------------------
            # SCOPED FALLBACK
            #
            # Questions such as "what is this document about?"
            # contain no distinctive keywords, so the full-text
            # predicate matches nothing. When the user has
            # explicitly scoped the chat to specific documents,
            # return the leading chunks of those documents
            # instead of no evidence at all.
            # ------------------------------------------------

            if not rows and document_filter_applied:

                cursor.execute(
                    build_sql(False),
                    [
                        query,
                        term_patterns,
                        term_patterns,
                        *condition_params,
                        limit,
                    ],
                )

                rows = cursor.fetchall()

            conn.rollback()

            return [
                dict(row)
                for row in rows
            ]

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # COUNT INDEXED CHUNKS
    # ========================================================

    def count_document_chunks(
        self,
        user_id,
        document_ids,
    ):

        clean_ids = []

        for value in document_ids or []:

            try:
                clean_ids.append(
                    int(value)
                )
            except Exception:
                continue

        if not clean_ids:
            return {}

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    d.id,
                    COUNT(c.id)

                FROM public.documents d

                LEFT JOIN public.document_chunks c
                    ON c.document_id = d.id

                WHERE d.user_id = %s
                AND d.id = ANY(%s)

                GROUP BY d.id
                """,
                (
                    int(user_id),
                    clean_ids,
                ),
            )

            counts = {
                int(row[0]): int(row[1])
                for row in cursor.fetchall()
            }

            conn.rollback()

            return counts

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    def create_conversation(
        self,
        user_id,
        title="New Conversation",
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id
                FROM public.users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            if not cursor.fetchone():

                raise RuntimeError(
                    "User no longer exists."
                )

            cursor.execute(
                """
                INSERT INTO public.conversations
                (
                    user_id,
                    title
                )
                VALUES
                (
                    %s,
                    %s
                )
                RETURNING id
                """,
                (
                    int(user_id),
                    str(title),
                ),
            )

            conversation_id = int(
                cursor.fetchone()[0]
            )

            conn.commit()

            return conversation_id

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # GET CONVERSATIONS
    # ========================================================

    def get_conversations(self, user_id):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT
                    id,
                    user_id,
                    title,
                    created_at
                FROM public.conversations
                WHERE user_id = %s
                ORDER BY
                    created_at DESC,
                    id DESC
                """,
                (int(user_id),),
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [
                dict(row)
                for row in rows
            ]

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # DELETE CONVERSATION
    # ========================================================

    def delete_conversation(
        self,
        conversation_id,
        user_id=None,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            if user_id is None:

                cursor.execute(
                    """
                    DELETE FROM public.conversations
                    WHERE id = %s
                    """,
                    (int(conversation_id),),
                )

            else:

                cursor.execute(
                    """
                    DELETE FROM public.conversations
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        int(conversation_id),
                        int(user_id),
                    ),
                )

            deleted = cursor.rowcount > 0

            conn.commit()

            return deleted

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # GET MESSAGES
    # ========================================================

    def get_messages(
        self,
        conversation_id,
        user_id=None,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            if user_id is None:

                cursor.execute(
                    """
                    SELECT
                        m.id,
                        m.conversation_id,
                        m.role,
                        m.content,
                        m.created_at
                    FROM public.messages m
                    WHERE m.conversation_id = %s
                    ORDER BY m.id ASC
                    """,
                    (int(conversation_id),),
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        m.id,
                        m.conversation_id,
                        m.role,
                        m.content,
                        m.created_at
                    FROM public.messages m
                    INNER JOIN public.conversations c
                        ON c.id = m.conversation_id
                    WHERE m.conversation_id = %s
                    AND c.user_id = %s
                    ORDER BY m.id ASC
                    """,
                    (
                        int(conversation_id),
                        int(user_id),
                    ),
                )

            rows = cursor.fetchall()

            conn.rollback()

            return [
                dict(row)
                for row in rows
            ]

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # SAVE MESSAGE
    # ========================================================

    def save_message(
        self,
        conversation_id,
        role,
        content,
        user_id=None,
    ):

        if not content:
            return

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            if user_id is not None:

                cursor.execute(
                    """
                    SELECT id
                    FROM public.conversations
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        int(conversation_id),
                        int(user_id),
                    ),
                )

                if not cursor.fetchone():

                    raise RuntimeError(
                        "Conversation does not belong "
                        "to the current user."
                    )

            cursor.execute(
                """
                INSERT INTO public.messages
                (
                    conversation_id,
                    role,
                    content
                )
                VALUES
                (
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    int(conversation_id),
                    str(role),
                    str(content),
                ),
            )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # UPDATE CONVERSATION TITLE
    # ========================================================

    def update_conversation_title(
        self,
        conversation_id,
        title,
        user_id=None,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            if user_id is None:

                cursor.execute(
                    """
                    UPDATE public.conversations
                    SET title = %s
                    WHERE id = %s
                    """,
                    (
                        str(title),
                        int(conversation_id),
                    ),
                )

            else:

                cursor.execute(
                    """
                    UPDATE public.conversations
                    SET title = %s
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        str(title),
                        int(conversation_id),
                        int(user_id),
                    ),
                )

            conn.commit()

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_stats(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            stats = {}

            for key, table in [
                ("users", "users"),
                ("documents", "documents"),
                ("chunks", "document_chunks"),
                ("conversations", "conversations"),
                ("messages", "messages"),
            ]:

                cursor.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM public.{table}
                    """
                )

                stats[key] = int(
                    cursor.fetchone()[0]
                )

            conn.rollback()

            return stats

        except Exception:

            conn.rollback()
            raise

        finally:

            if cursor:
                cursor.close()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text_from_image(file_bytes):

    # --------------------------------------------------------
    # Preferred path: the project's own OCR processor, which
    # also preprocesses the image before running Tesseract.
    # --------------------------------------------------------

    if OCRProcessor is not None:

        try:
            processor = OCRProcessor()

        except Exception:
            # PyMuPDF/pytesseract missing: fall back below.
            processor = None

        if processor is not None:

            try:
                return str(
                    processor.extract_text_from_bytes(
                        file_bytes
                    )
                    or ""
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Could not read image file: {exc}"
                ) from exc

    if Image is None:

        raise RuntimeError(
            "Image support requires Pillow. "
            "Install pillow."
        )

    if pytesseract is None:

        raise RuntimeError(
            "Image text extraction requires pytesseract. "
            "Install pytesseract and the Tesseract OCR "
            "engine."
        )

    try:

        with Image.open(
            io.BytesIO(file_bytes)
        ) as image:

            image.load()

            if image.mode not in {"L", "RGB"}:
                image = image.convert("RGB")

            text = pytesseract.image_to_string(image)

    except RuntimeError:
        raise

    except Exception as exc:

        raise RuntimeError(
            f"Could not read image file: {exc}"
        ) from exc

    return str(text or "")


def extract_document(
    filename,
    file_bytes,
):

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        if PdfReader is None:

            raise RuntimeError(
                "pypdf is not installed."
            )

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            if text.strip():

                pages.append(
                    {
                        "page": page_number,
                        "content": text,
                        "chunk_type": "Document",
                        "metadata": {
                            "page": page_number
                        },
                    }
                )

        return pages

    # ========================================================
    # DOCX
    # ========================================================

    if extension == ".docx":

        if DocxDocument is None:

            raise RuntimeError(
                "python-docx is not installed."
            )

        document = DocxDocument(
            io.BytesIO(file_bytes)
        )

        parts = []

        for paragraph in document.paragraphs:

            text = str(
                paragraph.text or ""
            ).strip()

            if text:
                parts.append(text)

        # Also include table content.
        for table_index, table in enumerate(
            document.tables,
            start=1,
        ):

            rows = []

            for row in table.rows:

                rows.append(
                    " | ".join(
                        str(
                            cell.text or ""
                        ).strip()
                        for cell in row.cells
                    )
                )

            if rows:

                parts.append(
                    "\n".join(
                        [
                            f"TABLE {table_index}",
                            *rows,
                        ]
                    )
                )

        content = "\n\n".join(parts)

        if not content.strip():
            return []

        return [
            {
                "page": None,
                "content": content,
                "chunk_type": "Document",
            }
        ]

    # ========================================================
    # PPTX
    # ========================================================

    if extension == ".pptx":

        if Presentation is None:

            raise RuntimeError(
                "python-pptx is not installed."
            )

        presentation = Presentation(
            io.BytesIO(file_bytes)
        )

        pages = []

        for slide_number, slide in enumerate(
            presentation.slides,
            start=1,
        ):

            parts = []

            for shape in slide.shapes:

                if hasattr(shape, "text"):

                    text = str(
                        shape.text or ""
                    ).strip()

                    if text:
                        parts.append(text)

            content = "\n".join(parts)

            if content.strip():

                pages.append(
                    {
                        "page": slide_number,
                        "content": content,
                        "chunk_type": "Document",
                        "metadata": {
                            "slide": slide_number
                        },
                    }
                )

        return pages

    # ========================================================
    # XLSX / XLS
    # ========================================================

    if extension in {".xlsx", ".xls"}:

        try:

            sheets = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=None,
            )

        except ImportError as exc:

            raise RuntimeError(
                "Excel support requires the appropriate "
                "pandas Excel engine. For .xlsx install "
                "openpyxl; for .xls install xlrd."
            ) from exc

        except Exception as exc:

            raise RuntimeError(
                f"Could not read Excel file: {exc}"
            ) from exc

        result = []

        for sheet_name, dataframe in sheets.items():

            dataframe = dataframe.fillna("")

            content = dataframe.to_csv(
                index=False
            )

            if not content.strip():
                continue

            result.append(
                {
                    "page": str(sheet_name),
                    "content": content,
                    "chunk_type": "Table",
                    "metadata": {
                        "sheet": str(sheet_name)
                    },
                }
            )

        return result

    # ========================================================
    # CSV
    # ========================================================

    if extension == ".csv":

        try:

            dataframe = pd.read_csv(
                io.BytesIO(file_bytes)
            )

        except Exception:

            # Fallback for non-standard CSV encoding.
            dataframe = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin-1",
            )

        dataframe = dataframe.fillna("")

        content = dataframe.to_csv(
            index=False
        )

        return [
            {
                "page": None,
                "content": content,
                "chunk_type": "Table",
            }
        ]

    # ========================================================
    # IMAGES (OCR)
    # ========================================================

    if extension in IMAGE_EXTENSIONS:

        text = extract_text_from_image(file_bytes)

        if text and text.strip():

            return [
                {
                    "page": 1,
                    "content": text.strip(),
                    "chunk_type": "Image",
                    "metadata": {
                        "page": 1,
                        "source_type": "image",
                    },
                }
            ]

        return []

    # ========================================================
    # TXT / MARKDOWN
    # ========================================================

    content = file_bytes.decode(
        "utf-8",
        errors="replace",
    )

    if not content.strip():
        return []

    return [
        {
            "page": None,
            "content": content,
            "chunk_type": (
                "Markdown"
                if extension in {
                    ".md",
                    ".markdown",
                }
                else "Document"
            ),
        }
    ]


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_document(
    extracted_pages,
    method,
    chunk_size,
    chunk_overlap,
    filename,
):

    if not extracted_pages:
        return []

    all_chunks = []

    for page_data in extracted_pages:

        if not isinstance(
            page_data,
            dict,
        ):
            continue

        content = str(
            page_data.get(
                "content",
                "",
            )
            or ""
        )

        if not content.strip():
            continue

        page = page_data.get("page")

        chunk_type = page_data.get(
            "chunk_type",
            "Document",
        )

        metadata = page_data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        tables = []

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        if chunk_type == "Table":

            try:

                dataframe = pd.read_csv(
                    io.StringIO(content)
                )

                dataframe = dataframe.fillna("")

                tables.append(
                    {
                        "table": dataframe.to_dict(
                            orient="records"
                        ),
                        "page": page,
                        "sheet": metadata.get(
                            "sheet"
                        ),
                        "source": filename,
                    }
                )

            except Exception:
                pass

        engine = ChunkingEngine(
            text=content,
            metadata={
                **metadata,
                "filename": filename,
            },
            tables=tables,
            source=filename,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            pages=[
                {
                    "page": page,
                    "text": content,
                }
            ],
        )

        try:

            # ------------------------------------------------
            # Tables must use table chunking.
            # ------------------------------------------------

            if chunk_type == "Table":

                chunks = engine.chunk(
                    "Table"
                )

            elif method == "Multimodal":

                chunks = engine.chunk(
                    "Multimodal"
                )

            else:

                chunks = engine.chunk(
                    method
                )

        except Exception as exc:

            st.warning(
                f"Chunking method '{method}' failed "
                f"for {filename}: {exc}. "
                "Falling back to Recursive."
            )

            chunks = engine.chunk(
                "Recursive"
            )

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk = dict(chunk)

            chunk["filename"] = filename

            if chunk.get("page") is None:
                chunk["page"] = page

            chunk.setdefault(
                "chunk_type",
                chunk_type,
            )

            # OCR output keeps its Image type so the chunk-type
            # filter in the chat sidebar can select it.
            if chunk_type == "Image":
                chunk["chunk_type"] = "Image"

            content_value = str(
                chunk.get(
                    "content",
                    "",
                )
                or ""
            )

            if not content_value.strip():
                continue

            chunk["content"] = content_value

            chunk.setdefault(
                "characters",
                len(content_value),
            )

            chunk.setdefault(
                "tokens",
                max(
                    1,
                    len(
                        content_value.split()
                    ),
                ),
            )

            chunk.setdefault(
                "metadata",
                {},
            )

            all_chunks.append(chunk)

    # --------------------------------------------------------
    # Global chunk IDs.
    # --------------------------------------------------------

    for index, chunk in enumerate(
        all_chunks,
        start=1,
    ):

        chunk["chunk_id"] = index

    return all_chunks


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def create_llm():

    if not GOOGLE_API_KEY:
        return None

    try:

        llm = GeminiModel()

        return llm

    except Exception:

        return None


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {
        "authenticated": False,
        "user": None,
        "db": None,
        "vector_db": None,
        "multi_document_retriever": None,
        "llm": None,
        "conversation_id": None,
        "selected_document_ids": [],
        "selected_chunk_types": [],
        "top_k": DEFAULT_TOP_K,
        "chunking_method": "Recursive",
        "document_search": "",
        "web_search_permission": False,
        "web_permission_request": None,
        "search_source_mode": "Documents → ask before Web",
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# ============================================================
# DATABASE STARTUP
# ============================================================

if st.session_state.db is None:

    try:

        db = PostgreSQLStore()

        db.initialize()

        st.session_state.db = db

    except Exception as exc:

        st.error(
            "❌ Database initialization failed."
        )

        st.exception(exc)

        st.markdown(
            """
### PostgreSQL configuration

Configure either:

`DATABASE_URL`

or:

- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
"""
        )

        st.stop()


# ============================================================
# VECTOR DATABASE / MULTI-DOCUMENT RETRIEVER
# ============================================================

if st.session_state.vector_db is None:
    try:
        vector_db = VectorDatabase()
        vector_db.load()

        st.session_state.vector_db = vector_db
        st.session_state.multi_document_retriever = (
            MultiDocumentRetriever(vector_db)
        )

    except Exception as exc:
        st.session_state.vector_db = None
        st.session_state.multi_document_retriever = None

        print(
            "MULTI-DOCUMENT RETRIEVER INITIALIZATION FAILED:",
            exc,
        )


# ============================================================
# USER VALIDATION
# ============================================================

def validate_logged_in_user():

    if not st.session_state.authenticated:
        return False

    user = st.session_state.user

    if not user:
        return False

    try:

        fresh_user = (
            st.session_state.db.get_user_by_id(
                user["id"]
            )
        )

        if not fresh_user:

            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.conversation_id = None
            st.session_state.selected_document_ids = []

            return False

        st.session_state.user = fresh_user

        return True

    except Exception:

        return False


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title(
        "📄 Enterprise Document Intelligence"
    )

    st.caption(
        "PostgreSQL • RAG • Google Gemini"
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

        with st.form(
            "login_form",
            clear_on_submit=False,
        ):

            username = st.text_input(
                "Username"
            )

            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

            if submitted:

                if not username or not password:

                    st.error(
                        "Enter username and password."
                    )

                else:

                    try:

                        user = (
                            st.session_state.db.authenticate(
                                username,
                                password,
                            )
                        )

                        if user:

                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.conversation_id = None
                            st.session_state.selected_document_ids = []
                            st.session_state.selected_chunk_types = []
                            st.session_state.llm = create_llm()

                            st.rerun()

                        else:

                            st.error(
                                "Invalid username or password."
                            )

                    except Exception as exc:

                        st.error(
                            "Login failed."
                        )

                        st.exception(exc)

    # ========================================================
    # REGISTER
    # ========================================================

    with register_tab:

        st.caption(
            "The first registered account becomes an administrator."
        )

        with st.form(
            "register_form",
            clear_on_submit=False,
        ):

            username = st.text_input(
                "New username"
            )

            password = st.text_input(
                "Password",
                type="password",
            )

            confirm = st.text_input(
                "Confirm password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )

            if submitted:

                if not username.strip():

                    st.error(
                        "Username is required."
                    )

                elif len(password) < 6:

                    st.error(
                        "Password must contain at least 6 characters."
                    )

                elif password != confirm:

                    st.error(
                        "Passwords do not match."
                    )

                else:

                    result = (
                        st.session_state.db.create_user(
                            username,
                            password,
                        )
                    )

                    if result:

                        role = result[2]

                        if role == "admin":

                            st.success(
                                "Account created. "
                                "This is the first account, "
                                "so it has administrator access."
                            )

                        else:

                            st.success(
                                "Account created. "
                                "You can now log in."
                            )

                    else:

                        st.error(
                            "Could not create account. "
                            "The username may already exist."
                        )


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.conversation_id = None
    st.session_state.llm = None
    st.session_state.selected_document_ids = []
    st.session_state.selected_chunk_types = []
    st.session_state.web_search_permission = False
    st.session_state.web_permission_request = None

    st.rerun()


# ============================================================
# CHAT HELPERS
# ============================================================

def create_new_conversation():

    user = st.session_state.user

    conversation_id = (
        st.session_state.db.create_conversation(
            user["id"],
            "New Conversation",
        )
    )

    st.session_state.conversation_id = (
        conversation_id
    )


def ensure_conversation():

    if st.session_state.conversation_id:
        return

    create_new_conversation()


def load_conversation_messages():

    conversation_id = (
        st.session_state.conversation_id
    )

    if not conversation_id:
        return []

    return (
        st.session_state.db.get_messages(
            conversation_id,
            user_id=st.session_state.user["id"],
        )
    )


# ============================================================
# RETRIEVAL
# ============================================================

def selected_document_ids():

    return [
        document_id
        for document_id
        in (
            st.session_state.selected_document_ids
            or []
        )
    ]


def _has_local_keyword_evidence(query, results):
    """Return True when stored evidence contains a meaningful query term."""
    import re
    stopwords = {
        "what", "is", "the", "a", "an", "of", "to", "in", "for",
        "and", "or", "on", "with", "this", "that", "are", "was",
        "were", "do", "does", "how", "why", "where", "who", "which"
    }
    terms = {x for x in re.findall(r"\b[a-zA-Z0-9]{2,}\b", str(query or "").lower()) if x not in stopwords}
    if not terms:
        return False
    for r in results or []:
        if isinstance(r, dict):
            text = str(r.get("content") or r.get("text") or "").lower()
            name = str(r.get("filename") or r.get("source") or "").lower()
            if any(x in text or x in name for x in terms):
                return True
    return False


def _is_broad_document_question(query):
    """Return True for questions that need more evidence than a normal top-k lookup.

    Counting/listing/comparison questions often require several chunks rather
    than only the single most similar chunk. This is query-general and does
    not depend on any document-specific keyword.
    """
    text = str(query or "").lower()
    broad_terms = (
        "how many", "number of", "total number", "total",
        "list all", "all tables", "tables are there", "which tables",
        "all sections", "all figures", "which sections",
        "compare", "difference between", "summarize", "summary of",
        "what are the", "how many tables", "count the tables",
    )
    return any(term in text for term in broad_terms)


def _contextualize_followup_query(query):
    """Resolve an obvious follow-up using the immediately preceding user turn."""
    text = str(query or "").strip()
    if not text:
        return text

    lower = text.lower()
    followup_markers = (
        "i mean", "i meant", "what i mean", "that", "those", "these",
        "it", "same", "again", "above", "previous", "earlier",
        "second time", "third time", "more specifically", "total",
        "also", "and what about", "what about", "how about",
    )
    looks_like_followup = (
        any(marker in lower for marker in followup_markers)
        or len(text.split()) <= 6
    )
    if not looks_like_followup:
        return text

    try:
        messages = load_conversation_messages()
    except Exception:
        messages = []

    # Use ONLY the immediately preceding user question. Pulling two or more
    # old questions into retrieval was causing unrelated documents to enter
    # structural questions such as table counts.
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        previous = str(message.get("content") or "").strip()
        if previous and previous.lower() != lower:
            return f"{previous}\nFollow-up clarification: {text}"

    return text


def _table_identity(item):
    """Return a stable identity for one actual table, not one table chunk."""
    import re

    filename = str(item.get("filename") or "Unknown").strip().lower()
    metadata = item.get("metadata") or {}
    chunk_data = item.get("chunk_data") or {}

    # Extraction pipelines may store the table label under different keys.
    label_keys = (
        "table_number", "table_no", "table_id", "table_name",
        "table_label", "table_title", "name", "title",
    )
    for data in (metadata, chunk_data):
        if isinstance(data, dict):
            for key in label_keys:
                value = data.get(key)
                if value is not None and str(value).strip():
                    label = str(value).strip().lower()
                    if "table" in label or re.fullmatch(r"[ivxlcdm]+", label):
                        return (filename, re.sub(r"\s+", " ", label))

    content = str(item.get("content") or item.get("text") or "").strip()
    # Prefer an explicit table caption/number embedded in the extracted text.
    match = re.search(
        r"\btable\s*(?:no\.?\s*)?([ivxlcdm]+|\d+)\b",
        content,
        flags=re.IGNORECASE,
    )
    if match:
        return (filename, f"table {match.group(1).lower()}")

    # If no label exists, identical/overlapping chunks from the same page
    # must not inflate the count. Page is the safest fallback identity.
    page = item.get("page")
    if page is not None and str(page).strip():
        return (filename, f"page {str(page).strip().lower()}")

    normalized = re.sub(r"\s+", " ", content).lower()
    return (filename, normalized[:500])


def _unique_table_results(table_results):
    """Deduplicate table chunks so each real table is counted once."""
    unique = []
    seen = set()
    for item in table_results or []:
        key = _table_identity(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _document_ids_matching_query(query, documents):
    """Find user documents explicitly named in the question."""
    import re
    q = str(query or "").lower()
    matches = []
    q_terms = set(re.findall(r"[a-z0-9]+", q))
    if not q_terms:
        return []

    for doc in documents or []:
        filename = str(doc.get("filename") or "").strip()
        if not filename:
            continue
        name_terms = set(re.findall(r"[a-z0-9]+", filename.lower()))
        # Ignore generic file-extension terms and very common question words.
        name_terms -= {"pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"}
        name_terms -= {"the", "paper", "document", "file"}
        if name_terms and len(name_terms & q_terms) >= 1:
            matches.append(doc)

    return matches


def _retrieve_all_table_chunks(document_ids=None, user_id=None):
    """Retrieve every indexed table chunk for the allowed document scope."""
    db = st.session_state.db
    user = st.session_state.user or {}
    uid = int(user_id if user_id is not None else user.get("id"))
    ids = []
    for value in document_ids or []:
        try:
            ids.append(int(value))
        except Exception:
            pass

    conn = db.connect()
    cursor = None
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        conditions = [
            "d.user_id = %s",
            "LOWER(COALESCE(c.chunk_type, '')) = 'table'",
        ]
        params = [uid]
        if ids:
            conditions.append("d.id = ANY(%s)")
            params.append(ids)

        cursor.execute(
            f"""
            SELECT c.id, c.document_id, c.chunk_id, c.chunk_type,
                   c.content, c.page, c.tokens, c.characters, c.metadata,
                   c.chunk_data, c.created_at, d.filename, d.file_type,
                   d.chunking_method
            FROM public.document_chunks c
            INNER JOIN public.documents d ON d.id = c.document_id
            WHERE {' AND '.join(conditions)}
            ORDER BY d.filename, c.page NULLS LAST, c.id
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]

        # PDF files in this application are stored as Document chunks;
        # they are not converted into chunk_type=Table rows.  Therefore a
        # structural table question must also inspect the document text for
        # actual table captions.  Do this only when no native Table chunks
        # were found, so XLSX/CSV/DOCX table handling remains unchanged.
        if not rows:
            fallback_conditions = [
                "d.user_id = %s",
                "COALESCE(c.content, '') <> ''",
            ]
            fallback_params = [uid]
            if ids:
                fallback_conditions.append("d.id = ANY(%s)")
                fallback_params.append(ids)

            cursor.execute(
                f"""
                SELECT c.id, c.document_id, c.chunk_id, c.chunk_type,
                       c.content, c.page, c.tokens, c.characters, c.metadata,
                       c.chunk_data, c.created_at, d.filename, d.file_type,
                       d.chunking_method
                FROM public.document_chunks c
                INNER JOIN public.documents d ON d.id = c.document_id
                WHERE {' AND '.join(fallback_conditions)}
                ORDER BY d.filename, c.page NULLS LAST, c.id
                """,
                fallback_params,
            )
            document_rows = [dict(row) for row in cursor.fetchall()]

            import re
            seen = set()
            extracted_tables = []
            caption_pattern = re.compile(
                r"^\s*(?:table|TABLE)\s*(?:no\.?\s*)?"
                r"([ivxlcdm]+|\d+)\s*[:.\-]?\s*(.*)$",
                re.IGNORECASE,
            )

            for row in document_rows:
                content = str(row.get("content") or "")
                for line in content.splitlines():
                    match = caption_pattern.match(line.strip())
                    if not match:
                        continue

                    number = match.group(1).upper()
                    key = (
                        str(row.get("document_id")),
                        number,
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    table_row = dict(row)
                    title = str(match.group(2) or "").strip()
                    table_row["content"] = (
                        f"Table {number}"
                        + (f": {title}" if title else "")
                    )
                    table_row["text"] = table_row["content"]
                    table_row["chunk_type"] = "Table"
                    table_row["retrieval_type"] = "document_table_caption"
                    extracted_tables.append(table_row)

            rows = extracted_tables

        conn.rollback()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()


def _table_question(query):
    text = str(query or "").lower()
    return "table" in text and any(
        x in text
        for x in (
            "how many", "number of", "total", "count",
            "which table", "list", "all tables", "tables are there",
        )
    )




def retrieve_documents(
    query,
    chunk_types=None,
    intent=None,
):
    """Retrieve from stored documents before any web fallback.

    Selected documents are searched through MultiDocumentRetriever.
    With no selection, PostgreSQL is used first because it is the
    authoritative user/document index; the VectorDatabase is then used
    as a semantic fallback. This prevents an empty document selection
    from accidentally becoming a web-search request.
    """

    user = st.session_state.user

    if not user or not query or not str(query).strip():
        return []

    original_query = str(query).strip()
    query = _contextualize_followup_query(original_query)

    document_ids = selected_document_ids()

    active_chunk_types = (
        chunk_types
        if chunk_types is not None
        else (
            st.session_state.selected_chunk_types
            or None
        )
    )

    try:
        top_k = int(st.session_state.top_k)
    except (TypeError, ValueError):
        top_k = 5

    if top_k <= 0:
        top_k = 5

    # Broad analytical questions need a larger evidence pool. The final
    # answer generator still receives a bounded context, but retrieval must
    # see enough chunks to answer counting/listing/summary questions.
    retrieval_top_k = max(top_k, 25) if _is_broad_document_question(query) else top_k

    # --------------------------------------------------------
    # STRUCTURAL TABLE QUESTIONS
    # --------------------------------------------------------
    # Do not try to answer a total-table question from top-k semantic
    # chunks. Retrieve every indexed table chunk for the allowed scope.
    if (intent == "table_search") or _table_question(query):
        try:
            table_scope_ids = document_ids or None

            # If the user names a document in the question (e.g.
            # "How many tables are in HYAMDNet?"), restrict the structural
            # count to that document. Otherwise the scope is all user documents.
            if not table_scope_ids:
                try:
                    is_admin = str(user.get("role", "user")).lower() == "admin"
                    docs = st.session_state.db.get_documents(
                        user_id=user["id"],
                        is_admin=is_admin,
                    ) or []
                    named_docs = _document_ids_matching_query(query, docs)
                    if named_docs:
                        table_scope_ids = [d.get("id") for d in named_docs]
                except Exception as exc:
                    print("TABLE DOCUMENT NAME MATCH FAILED:", exc)

            table_results = _retrieve_all_table_chunks(
                document_ids=table_scope_ids,
                user_id=user["id"],
            )

            if table_results:
                # A table may be split into multiple indexed chunks. Count
                # unique table identities, never raw chunk rows.
                unique_tables = _unique_table_results(table_results)

                by_document = {}
                for item in unique_tables:
                    name = str(item.get("filename") or "Unknown")
                    by_document.setdefault(name, 0)
                    by_document[name] += 1

                summary_lines = [
                    f"Actual unique table count: {len(unique_tables)}",
                    "Table count by document:",
                ]
                summary_lines.extend(
                    f"- {name}: {count} table(s)"
                    for name, count in by_document.items()
                )
                summary_lines.append("Unique tables found:")

                for index, item in enumerate(unique_tables, start=1):
                    label_match = __import__("re").search(
                        r"\btable\s*(?:no\.?\s*)?([ivxlcdm]+|\d+)\b",
                        str(item.get("content") or ""),
                        flags=__import__("re").IGNORECASE,
                    )
                    label = (
                        f"Table {label_match.group(1)}"
                        if label_match else f"Table {index}"
                    )
                    summary_lines.append(
                        f"{label} | {item.get('filename') or 'Unknown'} | "
                        f"Page: {item.get('page') if item.get('page') is not None else 'Unknown'}"
                    )

                summary = dict(unique_tables[0])
                summary["content"] = "\n".join(summary_lines)
                summary["text"] = summary["content"]
                summary["retrieval_type"] = "structural_table_summary"
                summary["table_count"] = len(unique_tables)
                summary["table_documents"] = by_document

                print(
                    "STRUCTURAL TABLE RETRIEVAL:",
                    original_query,
                    "| RAW TABLE CHUNKS:", len(table_results),
                    "| UNIQUE TABLES:", len(unique_tables),
                    "| SCOPE:", table_scope_ids or "ALL USER DOCUMENTS",
                )
                return [summary] + unique_tables

            # No indexed tables is meaningful evidence too. Do not fall back
            # to unrelated semantic chunks for a structural table question.
            return []

        except Exception as exc:
            print("STRUCTURAL TABLE RETRIEVAL FAILED:", exc)

    # --------------------------------------------------------
    # SELECTED DOCUMENTS
    # --------------------------------------------------------
    # Keep the working selected-document path unchanged: the
    # MultiDocumentRetriever receives the explicit document IDs.
    if document_ids:
        retriever = st.session_state.multi_document_retriever

        if retriever is not None:
            try:
                results = retriever.retrieve(
                    user_id=user["id"],
                    query=query,
                    selected_document_ids=document_ids,
                    chunk_types=active_chunk_types,
                    top_k=retrieval_top_k,
                )

                if results:
                    print(
                        "MULTI-DOCUMENT RETRIEVAL:",
                        query,
                        "| SELECTED DOCUMENTS:",
                        document_ids,
                        "| RETRIEVED CHUNKS:",
                        len(results),
                    )
                    return results

            except Exception as exc:
                print(
                    "SELECTED-DOCUMENT RETRIEVAL FAILED:",
                    exc,
                )

        # PostgreSQL remains the authoritative fallback for a
        # selected-document query.
        try:
            results = st.session_state.db.search_chunks(
                user_id=user["id"],
                query=query,
                selected_document_ids=document_ids,
                chunk_types=active_chunk_types,
                top_k=retrieval_top_k,
            )

            if results:
                return results

        except Exception as exc:
            print(
                "SELECTED-DOCUMENT POSTGRES RETRIEVAL FAILED:",
                exc,
            )

        return []

    # --------------------------------------------------------
    # NO DOCUMENT SELECTION
    # --------------------------------------------------------
    # No selection means ALL documents belonging to this user.
    # Run BOTH lexical PostgreSQL and semantic FAISS retrieval.
    # Exact terms (e.g. encoder, requirements.txt) are handled by
    # PostgreSQL; paraphrased questions are handled by FAISS.
    selected_sources = []
    try:
        is_admin = str(user.get("role", "user")).lower() == "admin"
        docs = st.session_state.db.get_documents(
            user_id=user["id"], is_admin=is_admin
        ) or []
        selected_sources = [
            str(d.get("filename", "")).strip()
            for d in docs
            if str(d.get("filename", "")).strip()
        ]
    except Exception as exc:
        print("ALL-DOCUMENT SCOPE LOOKUP FAILED:", exc)

    postgres_results = []
    try:
        postgres_results = st.session_state.db.search_chunks(
            user_id=user["id"],
            query=query,
            selected_document_ids=None,
            chunk_types=active_chunk_types,
            top_k=max(retrieval_top_k, 25),
        ) or []
        print("ALL-DOCUMENT POSTGRES RETRIEVAL:", query,
              "| RETRIEVED CHUNKS:", len(postgres_results))
    except Exception as exc:
        print("ALL-DOCUMENT POSTGRES RETRIEVAL FAILED:", exc)

    vector_results = []
    retriever = st.session_state.multi_document_retriever
    if retriever is not None and selected_sources:
        try:
            vector_results = retriever.retrieve(
                user_id=user["id"],
                query=query,
                selected_document_ids=None,
                selected_sources=selected_sources,
                chunk_types=active_chunk_types,
                top_k=max(retrieval_top_k, 25),
            ) or []
            print("ALL-DOCUMENT VECTOR RETRIEVAL:", query,
                  "| RETRIEVED CHUNKS:", len(vector_results))
        except Exception as exc:
            print("ALL-DOCUMENT VECTOR RETRIEVAL FAILED:", exc)

    # Merge both retrieval paths. Prefer lexical matches when a query term
    # occurs in the actual chunk; otherwise preserve semantic candidates.
    merged = []
    seen = set()
    for item in postgres_results + vector_results:
        if not isinstance(item, dict):
            continue
        key = (item.get("document_id"), item.get("chunk_id"), item.get("id"))
        if key == (None, None, None):
            key = str(item.get("content") or item.get("text") or "")[:250]
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    # Keep PostgreSQL lexical ranking ahead of weak semantic candidates.
    merged.sort(
        key=lambda x: float(
            x.get("similarity_score")
            if x.get("similarity_score") is not None
            else x.get("score") or 0
        ),
        reverse=True,
    )
    return merged[:top_k]





def selected_documents_unavailable_message():
    """Explain why a scoped retrieval produced no evidence."""

    document_ids = selected_document_ids()

    try:

        counts = (
            st.session_state.db.count_document_chunks(
                st.session_state.user["id"],
                document_ids,
            )
        )

    except Exception as exc:

        return (
            "I could not search the selected document(s) because the "
            f"document index could not be read: {exc}"
        )

    missing = []

    for document_id in document_ids:

        try:
            document_id = int(document_id)
        except Exception:
            continue

        if document_id not in counts:
            missing.append(document_id)

    if missing:

        return (
            "The selected document(s) are no longer available in the "
            "database. Please re-select or re-upload them."
        )

    unindexed = [
        document_id
        for document_id, count in counts.items()
        if count == 0
    ]

    if unindexed:

        return (
            "The document is selected, but it has no indexed content. "
            "Please re-index (re-upload) the document."
        )

    if st.session_state.selected_chunk_types:

        return (
            "The document is selected and indexed, but no chunks match "
            "the active chunk-type filter "
            f"({', '.join(st.session_state.selected_chunk_types)}). "
            "Clear the chunk-type filter and try again."
        )

    return (
        "The document is selected, but I could not retrieve its indexed "
        "content. Please re-index the document or check the document "
        "index."
    )


# ============================================================
# RAG
# ============================================================

def perform_rag(
    query,
    allow_web=False,
):

    llm = st.session_state.llm
    document_scope = selected_document_ids()

    print(
        "USER QUERY:",
        query,
        "| SELECTED DOCUMENTS:",
        document_scope,
    )

    # --------------------------------------------------------
    # No LLM.
    # --------------------------------------------------------
    if llm is None:
        local_results = retrieve_documents(query)
        if local_results:
            first = local_results[0]
            return (
                "The Gemini API key is not configured, but I found "
                "matching document evidence:\n\n"
                + str(first.get("content", "")),
                local_results,
                "documents",
            )
        return (
            "No usable evidence was found. Configure GOOGLE_API_KEY to "
            "enable AI-generated answers.",
            [],
            "error",
        )

    # --------------------------------------------------------
    # QUERY UNDERSTANDING FIRST
    # --------------------------------------------------------
    # Do not make retrieval depend on exact words such as
    # "how many tables". The planner is used as a semantic
    # query-understanding layer so natural, abbreviated, or
    # grammatically incorrect document questions can be mapped
    # to an appropriate retrieval intent/query.
    try:
        plan = llm.plan_action(
            query=query,
            previous_actions=[],
            previous_evaluations=[],
        ) or {}
    except Exception as exc:
        print("QUERY PLANNING FAILED:", exc)
        plan = {}

    action = str(plan.get("action", "vector_search") or "vector_search").strip()
    understood_query = str(plan.get("query", "") or "").strip()
    retrieval_query = understood_query or query

    # Never allow the planner's action to bypass local documents. A
    # web action is only executable after explicit user permission.
    planned_web = action == "web_search"
    if document_scope and planned_web:
        action = "vector_search"

    print(
        "QUERY UNDERSTANDING:",
        "| ACTION:", action,
        "| RETRIEVAL QUERY:", retrieval_query,
        "| REASON:", plan.get("reason"),
    )

    # --------------------------------------------------------
    # LOCAL DOCUMENT PREFLIGHT
    # --------------------------------------------------------
    # Always search stored documents before considering the web. Use
    # the semantically understood query, while retaining the original
    # wording as a second retrieval signal when it differs.
    local_results = retrieve_documents(
        retrieval_query,
        intent=("table_search" if action == "table_search" else None),
    )

    if retrieval_query.strip().lower() != str(query).strip().lower():
        try:
            original_results = retrieve_documents(
                query,
                intent=("table_search" if action == "table_search" else None),
            )
        except Exception as exc:
            print("ORIGINAL QUERY RETRIEVAL FAILED:", exc)
            original_results = []

        # Merge without allowing duplicate chunks to dominate the context.
        seen = set()
        merged = []
        for item in list(local_results or []) + list(original_results or []):
            if not isinstance(item, dict):
                continue
            key = (
                item.get("document_id"),
                item.get("id") or item.get("chunk_id"),
                item.get("page"),
                str(item.get("content") or item.get("text") or "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        local_results = merged

    if local_results:
        try:
            evaluation = llm.evaluate_evidence(
                query=query,
                results=local_results,
                action=action if action in {
                    "table_search", "document_search", "vector_search"
                } else "vector_search",
            ) or {}
        except Exception as exc:
            print("LOCAL EVIDENCE EVALUATION FAILED:", exc)
            evaluation = {"sufficient": False, "confidence": 0.0}

        if evaluation.get("sufficient", False):
            try:
                answer = llm.generate_answer(
                    query=query,
                    chunks=local_results,
                    source_type="uploaded documents",
                )
                return answer, local_results, "documents"
            except Exception as exc:
                print("LOCAL DOCUMENT ANSWER FAILED:", exc)

    local_fallback = local_results or []

    # --------------------------------------------------------
    # DOCUMENT SEARCH / TABLE SEARCH
    # --------------------------------------------------------
    if action in {"document_search", "table_search"}:
        # The first local preflight already used the planner's semantic
        # query and appropriate intent. If it was insufficient, do one
        # final local retrieval using the user's exact query before web.
        try:
            retry_results = retrieve_documents(
                query,
                intent=("table_search" if action == "table_search" else None),
            )
        except Exception:
            retry_results = []

        if retry_results:
            local_fallback = retry_results
            try:
                evaluation = llm.evaluate_evidence(
                    query=query,
                    results=retry_results,
                    action=action,
                ) or {}
            except Exception:
                evaluation = {"sufficient": False, "confidence": 0.0}

            if evaluation.get("sufficient", False):
                try:
                    answer = llm.generate_answer(
                        query=query,
                        chunks=retry_results,
                        source_type="uploaded tables" if action == "table_search" else "uploaded documents",
                    )
                    return answer, retry_results, "documents"
                except Exception as exc:
                    print("LOCAL ANSWER RETRY FAILED:", exc)

    # --------------------------------------------------------
    # WEB SEARCH -- EXPLICIT PERMISSION ONLY
    # --------------------------------------------------------
    if planned_web or action == "web_search":
        if not allow_web:
            return (
                "I couldn't find sufficient evidence in your stored documents. "
                "Would you like me to search the web?",
                local_fallback,
                "web_permission_required",
            )

        try:
            web_results = llm.web_search(
                retrieval_query or query,
                max_results=5,
            )
        except Exception as exc:
            print("WEB SEARCH FAILED:", exc)
            web_results = []

        if web_results:
            try:
                answer = llm.generate_answer(
                    query=query,
                    chunks=web_results,
                    source_type="web",
                )
                return answer, web_results, "web"
            except Exception as exc:
                print("WEB ANSWER FAILED:", exc)

    # --------------------------------------------------------
    # FINAL LOCAL FALLBACK
    # --------------------------------------------------------
    if local_fallback:
        try:
            answer = llm.generate_answer(
                query=query,
                chunks=local_fallback,
                source_type="uploaded documents",
            )
            return answer, local_fallback, "documents"
        except Exception as exc:
            print("FINAL LOCAL ANSWER FAILED:", exc)

    return (
        "I couldn't find sufficient evidence in your stored documents. "
        "Would you like me to search the web?",
        [],
        "web_permission_required",
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    user = st.session_state.user

    with st.sidebar:

        st.markdown(
            "## 📄 Enterprise RAG"
        )

        st.caption(
            f"Logged in as **{user['username']}**"
        )

        st.caption(
            f"Role: **{user['role']}**"
        )

        st.divider()

        pages = [
            "💬 Chat",
            "📚 Documents",
        ]

        if str(
            user["role"]
        ).lower() == "admin":

            pages.append(
                "🛠️ Admin"
            )

        page = st.radio(
            "Navigation",
            pages,
        )

        st.divider()

        st.markdown(
            "### Retrieval settings"
        )

        st.session_state.top_k = st.slider(
            "Top results",
            min_value=1,
            max_value=15,
            value=int(
                st.session_state.top_k
            ),
        )

        methods = [
            "Character",
            "Recursive",
            "Token",
            "Markdown",
            "Context",
            "Multimodal",
        ]

        current_method = (
            st.session_state.chunking_method
        )

        if current_method not in methods:
            current_method = "Recursive"

        st.session_state.chunking_method = (
            st.selectbox(
                "Default chunking method",
                methods,
                index=methods.index(
                    current_method
                ),
            )
        )

        st.divider()

        # --------------------------------------------------------
        # ANSWER SOURCE
        # --------------------------------------------------------
        source_options = [
            "Documents only",
            "Documents → ask before Web",
            "Web",
        ]

        current_source_mode = st.session_state.get(
            "search_source_mode",
            "Documents → ask before Web",
        )

        if current_source_mode not in source_options:
            current_source_mode = (
                "Documents → ask before Web"
            )

        st.session_state.search_source_mode = st.radio(
            "Answer source",
            source_options,
            index=source_options.index(
                current_source_mode
            ),
            help=(
                "Documents only: never search the web. "
                "Documents → ask before Web: search stored "
                "documents first and ask before web. "
                "Web: web search still requires explicit "
                "confirmation."
            ),
        )

        st.divider()

        if st.button(
            "➕ New conversation",
            use_container_width=True,
        ):

            create_new_conversation()
            st.rerun()

        if st.button(
            "🚪 Logout",
            use_container_width=True,
        ):

            logout()

        return page


# ============================================================
# CHAT PAGE
# ============================================================

def chat_page():

    st.title(
        "💬 Enterprise Document Assistant"
    )

    if st.session_state.llm is None:

        st.warning(
            "⚠️ The Gemini API is not configured. "
            "Set GOOGLE_API_KEY in your "
            "environment/configuration."
        )

    selected_ids = (
        st.session_state.selected_document_ids
    )

    if selected_ids:

        docs = []

        for document_id in selected_ids:

            try:

                document = (
                    st.session_state.db.get_document_by_id(
                        document_id,
                        user_id=st.session_state.user["id"],
                        is_admin=False,
                    )
                )

            except Exception:

                document = None

            if document:
                docs.append(document)

        if docs:

            st.info(
                "🎯 Searching only selected document(s): "
                + ", ".join(
                    d["filename"]
                    for d in docs
                )
            )

        else:

            st.session_state.selected_document_ids = []

    else:

        st.caption(
            "🔎 Chat searches across all your documents."
        )

    ensure_conversation()

    # --------------------------------------------------------
    # Conversations
    # --------------------------------------------------------

    try:

        conversations = (
            st.session_state.db.get_conversations(
                st.session_state.user["id"]
            )
        )

    except Exception as exc:

        st.error(
            "Could not load conversations."
        )

        st.exception(exc)

        return

    if conversations:

        ids = [
            item["id"]
            for item in conversations
        ]

        titles = [
            f"{item['title']} (#{item['id']})"
            for item in conversations
        ]

        current_index = 0

        if (
            st.session_state.conversation_id
            in ids
        ):

            current_index = ids.index(
                st.session_state.conversation_id
            )

        conv_col, delete_col = st.columns(
            [5, 1]
        )

        with conv_col:

            selected = st.selectbox(
                "Conversation",
                titles,
                index=current_index,
            )

        selected_index = titles.index(
            selected
        )

        selected_id = ids[
            selected_index
        ]

        with delete_col:

            st.write("")
            st.write("")

            if st.button(
                "🗑️ Delete",
                key=(
                    f"delete_conversation_"
                    f"{selected_id}"
                ),
                use_container_width=True,
            ):

                try:

                    deleted = (
                        st.session_state.db
                        .delete_conversation(
                            selected_id,
                            user_id=st.session_state.user["id"],
                        )
                    )

                    if deleted:

                        if (
                            st.session_state.conversation_id
                            == selected_id
                        ):

                            st.session_state.conversation_id = None

                        st.success(
                            "Conversation deleted."
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Could not delete conversation."
                        )

                except Exception as exc:

                    st.error(
                        f"Delete failed: {exc}"
                    )

        if (
            selected_id
            != st.session_state.conversation_id
        ):

            st.session_state.conversation_id = (
                selected_id
            )

            st.rerun()

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages = load_conversation_messages()

    for message in messages:

        role = message.get("role")

        if role not in {
            "user",
            "assistant",
        }:
            continue

        with st.chat_message(role):

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

    # --------------------------------------------------------
    # Pending web permission.
    pending_web = st.session_state.get(
        "web_permission_request"
    )

    if pending_web:
        st.warning(
            "No sufficient evidence was found in your "
            "stored documents."
        )

        st.markdown(
            "**Would you like me to search the web for this question?**"
        )

        web_yes, web_no = st.columns(2)

        with web_yes:
            if st.button(
                "🌐 Yes, Search the Web",
                key="approve_pending_web_search",
                use_container_width=True,
            ):
                approved_query = str(
                    pending_web.get(
                        "query",
                        "",
                    )
                ).strip()

                st.session_state.web_permission_request = None
                st.session_state.web_search_permission = True

                with st.spinner(
                    "Searching the web because you allowed it..."
                ):
                    try:
                        (
                            web_answer,
                            web_sources,
                            web_source_type,
                        ) = perform_rag(
                            approved_query,
                            allow_web=True,
                        )

                        st.session_state.web_search_permission = False

                        st.markdown(web_answer)

                        if web_sources:
                            with st.expander(
                                "🔎 Web evidence"
                            ):
                                st.caption(
                                    "Source type: web"
                                )

                                for index, source in enumerate(
                                    web_sources,
                                    start=1,
                                ):
                                    if not isinstance(
                                        source,
                                        dict,
                                    ):
                                        continue

                                    filename = (
                                        source.get("title")
                                        or source.get("source")
                                        or source.get("url")
                                        or "Web result"
                                    )

                                    st.markdown(
                                        f"**{index}. {filename}**"
                                    )

                                    url = source.get(
                                        "url"
                                    )

                                    if url:
                                        st.caption(
                                            f"URL: {url}"
                                        )

                                    content = (
                                        source.get("content")
                                        or source.get("text")
                                        or source.get("snippet")
                                        or ""
                                    )

                                    if content:
                                        st.code(
                                            str(content),
                                            language=None,
                                        )

                        try:
                            st.session_state.db.save_message(
                                conversation_id,
                                "assistant",
                                web_answer,
                                user_id=user_id,
                            )
                        except Exception:
                            pass

                    except Exception as exc:
                        st.session_state.web_search_permission = False
                        st.error(
                            f"Web search failed: {exc}"
                        )

        with web_no:
            if st.button(
                "📄 No, Stay With Documents",
                key="deny_pending_web_search",
                use_container_width=True,
            ):
                st.session_state.web_permission_request = None
                st.session_state.web_search_permission = False

                st.info(
                    "Okay. I will not search the web. "
                    "Your documents remain the only source."
                )

    # Prompt
    # --------------------------------------------------------

    prompt = st.chat_input(
        "Ask something about your documents..."
    )

    if not prompt:
        return

    prompt = prompt.strip()

    if not prompt:
        return

    conversation_id = (
        st.session_state.conversation_id
    )

    user_id = st.session_state.user["id"]

    # --------------------------------------------------------
    # Save user message.
    # --------------------------------------------------------

    try:

        st.session_state.db.save_message(
            conversation_id,
            "user",
            prompt,
            user_id=user_id,
        )

    except Exception as exc:

        st.error(
            f"Could not save your message: {exc}"
        )

        return

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # Generate answer.
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                (
                    answer,
                    sources,
                    source_type,
                ) = perform_rag(
                    prompt,
                    allow_web=False,
                )

                if source_type == "web_permission_required":
                    st.session_state.web_permission_request = {
                        "query": prompt,
                    }

                st.markdown(answer)

                # --------------------------------------------
                # Evidence
                # --------------------------------------------

                if (
                    sources
                    and source_type != "web_permission_required"
                ):

                    with st.expander(
                        "🔎 Retrieved evidence"
                    ):

                        st.caption(
                            f"Source type: {source_type}"
                        )

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            if not isinstance(
                                source,
                                dict,
                            ):
                                continue

                            filename = (
                                source.get(
                                    "filename"
                                )
                                or source.get(
                                    "source"
                                )
                                or "Unknown"
                            )

                            page = source.get(
                                "page"
                            )

                            # Normalize retrieval scores for display.
                            # FAISS and PostgreSQL use different score semantics,
                            # so never display raw scores as if they were the same.
                            similarity = source.get("similarity_score")
                            if similarity is None:
                                similarity = source.get("score")

                            try:
                                raw_score = float(similarity)
                                distance = source.get("distance")

                                if distance is not None:
                                    # FAISS L2 distance: lower is better.
                                    raw_distance = max(0.0, float(distance))
                                    display_relevance = 1.0 / (1.0 + raw_distance)
                                elif raw_score > 0.0:
                                    # PostgreSQL ts_rank: map the non-negative
                                    # rank into a bounded 0..1 display value.
                                    display_relevance = raw_score / (1.0 + raw_score)
                                else:
                                    # A PostgreSQL row can be returned through
                                    # the LIKE fallback with ts_rank == 0.0.
                                    # Derive a small lexical relevance score from
                                    # meaningful query terms so valid evidence is
                                    # not displayed as "0.000".
                                    import re
                                    stopwords = {
                                        "what", "is", "the", "a", "an", "of",
                                        "to", "in", "for", "and", "or", "on",
                                        "with", "this", "that", "are", "was",
                                        "were", "do", "does", "how", "why",
                                        "where", "who", "which"
                                    }
                                    query_terms = {
                                        term
                                        for term in re.findall(
                                            r"\b[a-zA-Z0-9]{2,}\b",
                                            str(prompt).lower(),
                                        )
                                        if term not in stopwords
                                    }
                                    source_text = str(
                                        source.get("content")
                                        or source.get("text")
                                        or ""
                                    ).lower()
                                    filename_text = str(filename).lower()
                                    matched_terms = sum(
                                        1
                                        for term in query_terms
                                        if term in source_text or term in filename_text
                                    )
                                    display_relevance = (
                                        matched_terms / len(query_terms)
                                        if query_terms
                                        else 0.0
                                    )

                                display_relevance = max(0.0, min(1.0, display_relevance))
                            except (TypeError, ValueError):
                                display_relevance = None

                            st.markdown(
                                f"**{index}. {filename}**"
                            )

                            if page is not None:

                                st.caption(
                                    f"Page/Sheet: {page}"
                                )

                            if similarity is not None:

                                try:

                                    if display_relevance is not None:
                                        st.caption(
                                            "Relevance: "
                                            f"{display_relevance:.3f}"
                                        )

                                except Exception:
                                    pass

                            url = source.get(
                                "url"
                            )

                            if url:

                                st.caption(
                                    f"URL: {url}"
                                )

                            content = (
                                source.get(
                                    "content"
                                )
                                or source.get(
                                    "text"
                                )
                                or ""
                            )

                            chunk_type = source.get(
                                "chunk_type"
                            )

                            rendered_as_table = False

                            if content and chunk_type == "Table":

                                try:

                                    table_df = pd.read_csv(
                                        io.StringIO(
                                            str(content)
                                        )
                                    )

                                    st.dataframe(
                                        table_df,
                                        use_container_width=True,
                                    )

                                    rendered_as_table = True

                                except Exception:

                                    rendered_as_table = False

                            if content and not rendered_as_table:

                                # st.code provides a built-in copy button and a
                                # scrollable container, which is much more useful
                                # for long retrieved/web evidence than a fixed
                                # text area.
                                st.code(
                                    str(content),
                                    language=None,
                                )

                            st.divider()

                # --------------------------------------------
                # Save assistant message.
                # --------------------------------------------

                st.session_state.db.save_message(
                    conversation_id,
                    "assistant",
                    answer,
                    user_id=user_id,
                )

                # --------------------------------------------
                # Auto-title conversation.
                # --------------------------------------------

                current_messages = (
                    st.session_state.db.get_messages(
                        conversation_id,
                        user_id=user_id,
                    )
                )

                if len(current_messages) <= 2:

                    title = prompt[:70]

                    if len(prompt) > 70:
                        title += "..."

                    st.session_state.db.update_conversation_title(
                        conversation_id,
                        title,
                        user_id=user_id,
                    )

            except Exception as exc:

                answer = (
                    "Something went wrong while "
                    "processing your request."
                )

                st.error(answer)
                st.exception(exc)

                try:

                    st.session_state.db.save_message(
                        conversation_id,
                        "assistant",
                        f"{answer}\n\n{exc}",
                        user_id=user_id,
                    )

                except Exception:
                    pass


# ============================================================
# DOCUMENT PAGE
# ============================================================

def documents_page():

    st.title(
        "📚 Document Management"
    )

    user = st.session_state.user

    is_admin = (
        str(
            user["role"]
        ).lower()
        == "admin"
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader(
        "Upload documents"
    )

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=[
            item.lstrip(".")
            for item in sorted(
                SUPPORTED_EXTENSIONS
            )
        ],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        chunk_method = st.selectbox(
            "Chunking method",
            [
                "Automatic",
                "Character",
                "Recursive",
                "Token",
                "Markdown",
                "Context",
                "Table",
                "Image",
                "Visual",
                "Multimodal",
            ],
            index=0,
            help=(
                "Automatic is recommended for most users. It analyzes "
                "the uploaded document and selects the most appropriate "
                "existing chunking method. Choose a method manually for "
                "advanced control."
            ),
        )

    with col2:

        chunk_size = st.number_input(
            "Chunk size",
            min_value=100,
            max_value=10000,
            value=int(
                DEFAULT_CHUNK_SIZE
            ),
            step=100,
        )

    with col3:

        max_overlap = max(
            0,
            int(chunk_size) - 1,
        )

        default_overlap = min(
            int(DEFAULT_CHUNK_OVERLAP),
            max_overlap,
        )

        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=max_overlap,
            value=default_overlap,
            step=(
                50
                if max_overlap >= 50
                else 1
            ),
        )

    # ========================================================
    # PROCESS UPLOAD
    # ========================================================

    if uploaded_files:

        if st.button(
            "🚀 Process and upload",
            type="primary",
            use_container_width=True,
        ):

            # ------------------------------------------------
            # Refresh user.
            # ------------------------------------------------

            fresh_user = (
                st.session_state.db.get_user_by_id(
                    user["id"]
                )
            )

            if not fresh_user:

                st.error(
                    "Your account no longer exists. "
                    "Please log in again."
                )

                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.conversation_id = None
                st.session_state.selected_document_ids = []

                st.rerun()

            st.session_state.user = fresh_user

            user = fresh_user

            upload_success = False

            # ------------------------------------------------
            # Process each file.
            # ------------------------------------------------

            for uploaded_file in uploaded_files:

                filename = (
                    uploaded_file.name
                )

                try:

                    file_bytes = (
                        uploaded_file.getvalue()
                    )

                    extension = Path(
                        filename
                    ).suffix.lower()

                    if extension not in SUPPORTED_EXTENSIONS:

                        st.error(
                            f"{filename}: unsupported file type."
                        )

                        continue

                    with st.status(
                        f"Processing {filename}...",
                        expanded=True,
                    ):

                        # ------------------------------------
                        # Extraction
                        # ------------------------------------

                        st.write(
                            "1️⃣ Extracting document..."
                        )

                        extracted = extract_document(
                            filename,
                            file_bytes,
                        )

                        if not extracted:

                            if extension in IMAGE_EXTENSIONS:

                                raise ValueError(
                                    "No readable text was found in "
                                    "this image."
                                )

                            raise ValueError(
                                "No text or data could be extracted."
                            )

                        st.write(
                            f"Extracted {len(extracted)} section(s)."
                        )

                        # ------------------------------------
                        # Chunking
                        # ------------------------------------

                        st.write(
                            "2️⃣ Creating chunks..."
                        )

                        # Automatic mode analyzes the extracted document
                        # and selects one of the existing chunking methods.
                        # Manual methods continue to work unchanged.
                        selected_chunk_method = chunk_method

                        if chunk_method == "Automatic":

                            table_sections = [
                                page_data
                                for page_data in extracted
                                if isinstance(page_data, dict)
                                and page_data.get("chunk_type") == "Table"
                            ]

                            image_sections = [
                                page_data
                                for page_data in extracted
                                if isinstance(page_data, dict)
                                and page_data.get("chunk_type") == "Image"
                            ]

                            visual_sections = [
                                page_data
                                for page_data in extracted
                                if isinstance(page_data, dict)
                                and page_data.get("chunk_type") in {
                                    "Visual",
                                    "Diagram",
                                    "Chart",
                                    "Flowchart",
                                }
                            ]

                            extracted_text = "\n".join(
                                str(
                                    page_data.get(
                                        "content",
                                        "",
                                    )
                                    or ""
                                )
                                for page_data in extracted
                                if isinstance(page_data, dict)
                            )

                            automatic_result = (
                                get_automatic_chunking_method(
                                    file_name=filename,
                                    file_type=extension,
                                    text=extracted_text,
                                    tables=table_sections,
                                    images=image_sections,
                                    visuals=visual_sections,
                                    pages=extracted,
                                )
                            )

                            selected_chunk_method = automatic_result

                            st.info(
                                f"🤖 Automatic chunking selected: "
                                f"**{selected_chunk_method}**"
                            )

                        chunks = chunk_document(
                            extracted,
                            selected_chunk_method,
                            int(chunk_size),
                            int(chunk_overlap),
                            filename,
                        )

                        if not chunks:

                            raise ValueError(
                                "Chunking produced zero usable chunks."
                            )

                        st.write(
                            f"Created {len(chunks)} chunk(s)."
                        )

                        # ------------------------------------
                        # Save
                        # ------------------------------------

                        st.write(
                            "3️⃣ Saving document and chunks..."
                        )

                        result = (
                            st.session_state.db
                            .save_document_with_chunks(
                                user_id=int(user["id"]),
                                filename=filename,
                                file_type=extension,
                                file_bytes=file_bytes,
                                chunking_method=selected_chunk_method,
                                chunk_size=int(chunk_size),
                                chunk_overlap=int(chunk_overlap),
                                chunks=chunks,
                                metadata={
                                    "original_filename": filename,
                                    "sections": len(extracted),
                                    "chunks": len(chunks),
                                    "chunking_method": selected_chunk_method,
                                    "chunking_mode": chunk_method,
                                },
                            )
                        )

                        st.write(
                            "4️⃣ PostgreSQL transaction committed."
                        )

                        st.write(
                            f"Document ID: "
                            f"{result['document_id']}"
                        )

                        st.write(
                            f"Chunks saved: "
                            f"{result['chunk_count']}"
                        )

                    st.success(
                        f"✅ {filename} uploaded and indexed successfully."
                    )

                    upload_success = True

                except Exception as exc:

                    st.error(
                        f"❌ Failed to process "
                        f"{filename}: {exc}"
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.exception(exc)

            if upload_success:

                st.session_state.selected_document_ids = []
                st.session_state.document_search = ""

                st.rerun()

    st.divider()

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

    st.subheader(
        "🔎 Find a document"
    )

    search_col, clear_col = st.columns(
        [5, 1]
    )

    with search_col:

        document_search = st.text_input(
            "Search by filename",
            value=(
                st.session_state.document_search
            ),
            placeholder=(
                "Example: report.pdf..."
            ),
            label_visibility="collapsed",
        )

        st.session_state.document_search = (
            document_search
        )

    with clear_col:

        if st.button(
            "Clear",
            use_container_width=True,
        ):

            st.session_state.document_search = ""

            st.rerun()

    # ========================================================
    # GET DOCUMENTS
    # ========================================================

    try:

        documents = (
            st.session_state.db.get_documents(
                user_id=int(user["id"]),
                is_admin=is_admin,
                search=document_search,
            )
        )

    except Exception as exc:

        st.error(
            "Could not load documents."
        )

        st.exception(exc)

        return

    if is_admin:

        st.subheader(
            f"All documents ({len(documents)})"
        )

    else:

        st.subheader(
            f"Your documents ({len(documents)})"
        )

    if not documents:

        if document_search:

            st.info(
                f'No documents match "{document_search}".'
            )

        else:

            st.info(
                "No documents have been uploaded yet."
            )

        return

    # ========================================================
    # RETRIEVAL DOCUMENT SELECTION
    # ========================================================

    st.subheader(
        "🎯 Retrieval document selection"
    )

    document_options = {
        f"{doc['filename']}  (#{doc['id']})":
        doc["id"]
        for doc in documents
    }

    valid_ids = set(
        document_options.values()
    )

    cleaned_selected_ids = []

    for document_id in (
        st.session_state.selected_document_ids
    ):

        try:

            document_id = int(
                document_id
            )

        except Exception:

            continue

        if document_id in valid_ids:
            cleaned_selected_ids.append(
                document_id
            )

    st.session_state.selected_document_ids = (
        cleaned_selected_ids
    )

    selected_labels_default = [
        label
        for label, document_id
        in document_options.items()
        if document_id
        in st.session_state.selected_document_ids
    ]

    selected_labels = st.multiselect(
        "Choose documents that Chat should search",
        options=list(
            document_options.keys()
        ),
        default=selected_labels_default,
        help=(
            "Select one or more documents. "
            "Chat will search only those documents. "
            "Leave empty to search all documents."
        ),
    )

    st.session_state.selected_document_ids = [
        document_options[label]
        for label in selected_labels
    ]

    if selected_labels:

        st.success(
            "🎯 Chat is restricted to: "
            + ", ".join(
                selected_labels
            )
        )

    else:

        st.caption(
            "No document filter is active. "
            "Chat searches across all your documents."
        )

    # ========================================================
    # CHUNK TYPE FILTER
    # ========================================================

    chunk_type_options = [
        "Recursive",
        "Character",
        "Token",
        "Markdown",
        "Context",
        "Table",
        "Image",
        "Visual",
    ]

    selected_chunk_types = st.multiselect(
        "Optional chunk-type filter",
        chunk_type_options,
        default=(
            st.session_state.selected_chunk_types
        ),
        help=(
            "Leave empty to search all chunk types."
        ),
    )

    st.session_state.selected_chunk_types = (
        selected_chunk_types
    )

    # ========================================================
    # DOCUMENT LIST
    # ========================================================

    st.subheader(
        "📄 Uploaded documents"
    )

    for document in documents:

        document_id = document["id"]
        filename = document["filename"]

        size = document.get(
            "file_size",
            0,
        )

        try:

            size_mb = (
                float(size or 0)
                / 1024
                / 1024
            )

        except Exception:

            size_mb = 0.0

        chunk_count = document.get(
            "chunk_count",
            0,
        )

        is_selected = (
            document_id
            in st.session_state.selected_document_ids
        )

        with st.container(
            border=True
        ):

            col1, col2, col3 = st.columns(
                [6, 2, 1]
            )

            with col1:

                prefix = (
                    "🎯 📄"
                    if is_selected
                    else "📄"
                )

                st.markdown(
                    f"### {prefix} {filename}"
                )

                if is_admin:

                    st.caption(
                        "Owner: "
                        f"{document.get('username', 'Unknown')}"
                    )

                st.caption(
                    f"ID: #{document_id} "
                    f"• Type: {document.get('file_type', '')} "
                    f"• Size: {size_mb:.2f} MB"
                )

                st.caption(
                    f"Chunks: {chunk_count} "
                    f"• Chunking: "
                    f"{document.get('chunking_method', '')}"
                )

            with col2:

                st.caption(
                    "Created:"
                )

                st.caption(
                    str(
                        document.get(
                            "created_at",
                            "",
                        )
                    )
                )

            with col3:

                if st.button(
                    "🗑️ Delete",
                    key=(
                        f"delete_doc_"
                        f"{document_id}"
                    ),
                ):

                    try:

                        deleted = (
                            st.session_state.db
                            .delete_document(
                                document_id,
                                user_id=user["id"],
                                is_admin=is_admin,
                            )
                        )

                        if deleted:

                            if (
                                document_id
                                in st.session_state.selected_document_ids
                            ):

                                st.session_state.selected_document_ids.remove(
                                    document_id
                                )

                            st.success(
                                "Document deleted."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Could not delete document."
                            )

                    except Exception as exc:

                        st.error(
                            f"Delete failed: {exc}"
                        )


# ============================================================
# ADMIN PAGE
# ============================================================

def admin_page():

    user = st.session_state.user

    if str(
        user["role"]
    ).lower() != "admin":

        st.error(
            "You do not have administrator access."
        )

        return

    st.title(
        "🛠️ Administration"
    )

    # ========================================================
    # STATS
    # ========================================================

    try:

        stats = (
            st.session_state.db.get_stats()
        )

    except Exception as exc:

        st.error(
            "Could not load statistics."
        )

        st.exception(exc)

        stats = {}

    columns = st.columns(5)

    for column, (
        label,
        key,
    ) in zip(
        columns,
        [
            ("Users", "users"),
            ("Documents", "documents"),
            ("Chunks", "chunks"),
            ("Chats", "conversations"),
            ("Messages", "messages"),
        ],
    ):

        with column:

            st.metric(
                label,
                stats.get(
                    key,
                    0,
                ),
            )

    st.divider()

    # ========================================================
    # USER MANAGEMENT
    # ========================================================

    st.subheader(
        "User management"
    )

    try:

        users = (
            st.session_state.db.get_users()
        )

    except Exception as exc:

        st.error(
            "Could not load users."
        )

        st.exception(exc)

        return

    for target in users:

        target_id = target["id"]

        target_role = str(
            target.get(
                "role",
                "user",
            )
        ).lower()

        with st.container(
            border=True
        ):

            col1, col2, col3, col4 = (
                st.columns(
                    [3, 2, 2, 2]
                )
            )

            with col1:

                st.markdown(
                    f"**{target['username']}**"
                )

                st.caption(
                    f"User ID: {target_id}"
                )

            with col2:

                st.write(
                    f"Role: `{target_role}`"
                )

            with col3:

                new_role = st.selectbox(
                    "Role",
                    [
                        "user",
                        "admin",
                    ],
                    index=(
                        1
                        if target_role == "admin"
                        else 0
                    ),
                    key=(
                        f"role_"
                        f"{target_id}"
                    ),
                )

                if (
                    new_role
                    != target_role
                ):

                    if st.button(
                        "Update role",
                        key=(
                            f"update_role_"
                            f"{target_id}"
                        ),
                    ):

                        try:

                            changed = (
                                st.session_state.db
                                .set_user_role(
                                    target_id,
                                    new_role,
                                )
                            )

                            if changed:

                                if (
                                    target_id
                                    == user["id"]
                                ):

                                    fresh = (
                                        st.session_state.db
                                        .get_user_by_id(
                                            user["id"]
                                        )
                                    )

                                    if fresh:

                                        st.session_state.user = (
                                            fresh
                                        )

                                st.success(
                                    "Role updated."
                                )

                                st.rerun()

                            else:

                                st.error(
                                    "Cannot remove the last administrator."
                                )

                        except Exception as exc:

                            st.error(
                                str(exc)
                            )

            with col4:

                with st.expander(
                    "Password"
                ):

                    new_password = st.text_input(
                        "New password",
                        type="password",
                        key=(
                            f"password_"
                            f"{target_id}"
                        ),
                    )

                    if st.button(
                        "Reset password",
                        key=(
                            f"reset_"
                            f"{target_id}"
                        ),
                    ):

                        if len(
                            new_password
                        ) < 6:

                            st.error(
                                "Password must be at least 6 characters."
                            )

                        else:

                            try:

                                (
                                    st.session_state.db
                                    .reset_user_password(
                                        target_id,
                                        new_password,
                                    )
                                )

                                st.success(
                                    "Password reset."
                                )

                            except Exception as exc:

                                st.error(
                                    str(exc)
                                )

                if target_id != user["id"]:

                    if st.button(
                        "Delete user",
                        key=(
                            f"delete_user_"
                            f"{target_id}"
                        ),
                    ):

                        try:

                            deleted = (
                                st.session_state.db
                                .delete_user(
                                    target_id
                                )
                            )

                            if deleted:

                                st.success(
                                    "User deleted."
                                )

                                st.rerun()

                            else:

                                st.error(
                                    "Could not delete user. "
                                    "The last administrator cannot be deleted."
                                )

                        except Exception as exc:

                            st.error(
                                str(exc)
                            )


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.authenticated:

    login_page()

    st.stop()


# ============================================================
# VALIDATE SESSION AGAINST DATABASE
# ============================================================

if not validate_logged_in_user():

    st.warning(
        "Your login session is no longer valid. "
        "Please log in again."
    )

    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.conversation_id = None
    st.session_state.selected_document_ids = []

    st.stop()


# ============================================================
# INITIALIZE LLM
# ============================================================

if st.session_state.llm is None:

    st.session_state.llm = create_llm()


# ============================================================
# ROUTING
# ============================================================

page = render_sidebar()

if page == "💬 Chat":

    chat_page()

elif page == "📚 Documents":

    documents_page()

elif page == "🛠️ Admin":

    admin_page()
