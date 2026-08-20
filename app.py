import io
import os
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from chunking import ChunkingEngine
from model import GeminiModel


# ============================================================
# OPTIONAL POSTGRESQL DRIVER
# ============================================================

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import (
        RealDictCursor,
        Json,
    )
except ImportError:
    psycopg2 = None
    sql = None
    RealDictCursor = None
    Json = None


try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None


# ============================================================
# OPTIONAL DOCUMENT LIBRARIES
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
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None


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
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("POSTGRESQL_URL", ""),
)

PG_HOST = os.getenv(
    "PG_HOST",
    "localhost",
)

PG_PORT = os.getenv(
    "PG_PORT",
    "5432",
)

PG_DATABASE = os.getenv(
    "PG_DATABASE",
    "enterprise_rag",
)

PG_USER = os.getenv(
    "PG_USER",
    "postgres",
)

PG_PASSWORD = os.getenv(
    "PG_PASSWORD",
    "postgres",
)


# ============================================================
# ADMIN / PGVECTOR CONFIGURATION
# ============================================================

ADMIN_USERNAME = os.getenv(
    "ADMIN_USERNAME",
    "",
).strip()


PGVECTOR_ENABLED = os.getenv(
    "PGVECTOR_ENABLED",
    "true",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


PGVECTOR_DIMENSION = int(
    os.getenv(
        "PGVECTOR_DIMENSION",
        "768",
    )
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001",
)


# ============================================================
# APPLICATION DEFAULTS
# ============================================================

DEFAULT_CHUNK_SIZE = 800

DEFAULT_CHUNK_OVERLAP = 120

DEFAULT_TOP_K = 5


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".docx",
    ".xlsx",
    ".xls",
}


# ============================================================
# POSTGRESQL DATABASE
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
                "Add psycopg2-binary to requirements.txt."
            )

        if self.connection is not None:

            try:

                if self.connection.closed == 0:

                    try:
                        self.connection.rollback()
                    except Exception:
                        pass

                    return self.connection

            except Exception:

                try:
                    self.connection.close()
                except Exception:
                    pass

                self.connection = None

        if DATABASE_URL:

            self.connection = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=15,
            )

        else:

            self.connection = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD,
                connect_timeout=15,
            )

        self.connection.autocommit = False

        if register_vector is not None:

            try:

                register_vector(
                    self.connection
                )

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

    # ========================================================
    # ROLLBACK
    # ========================================================

    def rollback(self):

        if self.connection is not None:

            try:

                self.connection.rollback()

            except Exception:

                pass

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.connection is not None:

            try:

                self.connection.close()

            except Exception:

                pass

            self.connection = None

    # ========================================================
    # TABLE EXISTS
    # ========================================================

    def table_exists(
        self,
        cursor,
        table_name,
    ):

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
            """,
            (
                table_name,
            ),
        )

        return bool(
            cursor.fetchone()[0]
        )

    # ========================================================
    # COLUMN EXISTS
    # ========================================================

    def column_exists(
        self,
        cursor,
        table_name,
        column_name,
    ):

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                AND column_name = %s
            )
            """,
            (
                table_name,
                column_name,
            ),
        )

        return bool(
            cursor.fetchone()[0]
        )

    # ========================================================
    # ADD COLUMN
    # ========================================================

    def add_column_if_missing(
        self,
        cursor,
        table_name,
        column_name,
        definition,
    ):

        query = sql.SQL(
            """
            ALTER TABLE public.{}
            ADD COLUMN IF NOT EXISTS {} {}
            """
        ).format(
            sql.Identifier(table_name),
            sql.Identifier(column_name),
            sql.SQL(definition),
        )

        cursor.execute(query)

    # ========================================================
    # DROP CONSTRAINT
    # ========================================================

    def remove_constraint_if_exists(
        self,
        cursor,
        table_name,
        constraint_name,
    ):

        query = sql.SQL(
            """
            ALTER TABLE public.{}
            DROP CONSTRAINT IF EXISTS {}
            """
        ).format(
            sql.Identifier(table_name),
            sql.Identifier(constraint_name),
        )

        cursor.execute(query)

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def initialize(self):

        conn = self.connect()

        cursor = None

        try:

            conn.rollback()

            cursor = conn.cursor()

            cursor.execute(
                "SET search_path TO public"
            )

            # ==================================================
            # USERS
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255),
                    password_hash TEXT,
                    role VARCHAR(50) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

            self.add_column_if_missing(
                cursor,
                "users",
                "username",
                "VARCHAR(255)",
            )

            self.add_column_if_missing(
                cursor,
                "users",
                "password_hash",
                "TEXT",
            )

            self.add_column_if_missing(
                cursor,
                "users",
                "role",
                "VARCHAR(50) DEFAULT 'user'",
            )

            self.add_column_if_missing(
                cursor,
                "users",
                "created_at",
                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            )

            cursor.execute(
                """
                UPDATE public.users
                SET role = 'user'
                WHERE role IS NULL
                OR LOWER(role) NOT IN ('admin', 'user')
                """
            )

            conn.commit()

            try:

                cursor.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    users_username_unique_idx
                    ON public.users(username)
                    """
                )

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # DOCUMENTS
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    filename TEXT,
                    file_type TEXT,
                    file_size BIGINT,
                    file_data BYTEA,
                    chunking_method TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

            document_columns = {

                "user_id":
                    "INTEGER",

                "filename":
                    "TEXT",

                "file_type":
                    "TEXT",

                "file_size":
                    "BIGINT",

                "file_data":
                    "BYTEA",

                "chunking_method":
                    "TEXT",

                "chunk_size":
                    "INTEGER",

                "chunk_overlap":
                    "INTEGER",

                "metadata":
                    "JSONB DEFAULT '{}'::jsonb",

                "created_at":
                    "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for column_name, definition in document_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "documents",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # DOCUMENT CHUNKS
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER,
                    chunk_id INTEGER,
                    chunk_type TEXT,
                    content TEXT,
                    page INTEGER,
                    tokens INTEGER,
                    characters INTEGER,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    chunk_data JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

            chunk_columns = {

                "document_id":
                    "INTEGER",

                "chunk_id":
                    "INTEGER",

                "chunk_type":
                    "TEXT",

                "content":
                    "TEXT",

                "page":
                    "INTEGER",

                "tokens":
                    "INTEGER",

                "characters":
                    "INTEGER",

                "metadata":
                    "JSONB DEFAULT '{}'::jsonb",

                "chunk_data":
                    "JSONB DEFAULT '{}'::jsonb",

                "created_at":
                    "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for column_name, definition in chunk_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "document_chunks",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # PGVECTOR
            # ==================================================

            if PGVECTOR_ENABLED:

                try:

                    cursor.execute(
                        """
                        CREATE EXTENSION IF NOT EXISTS vector
                        """
                    )

                    conn.commit()

                    self.add_column_if_missing(
                        cursor,
                        "document_chunks",
                        "embedding",
                        f"vector({PGVECTOR_DIMENSION})",
                    )

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # CONVERSATIONS
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.conversations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

            # ==================================================
            # MESSAGES
            # ==================================================

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

            # ==================================================
            # FOREIGN KEYS
            # ==================================================

            constraints = [

                (
                    "documents",
                    "documents_user_id_fkey",
                    """
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                    """,
                ),

                (
                    "document_chunks",
                    "document_chunks_document_id_fkey",
                    """
                    FOREIGN KEY (document_id)
                    REFERENCES public.documents(id)
                    ON DELETE CASCADE
                    """,
                ),

                (
                    "conversations",
                    "conversations_user_id_fkey",
                    """
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                    """,
                ),

                (
                    "messages",
                    "messages_conversation_id_fkey",
                    """
                    FOREIGN KEY (conversation_id)
                    REFERENCES public.conversations(id)
                    ON DELETE CASCADE
                    """,
                ),
            ]

            for table_name, constraint_name, definition in constraints:

                try:

                    query = sql.SQL(
                        """
                        ALTER TABLE public.{}
                        ADD CONSTRAINT {}
                        {}
                        """
                    ).format(
                        sql.Identifier(table_name),
                        sql.Identifier(constraint_name),
                        sql.SQL(definition),
                    )

                    cursor.execute(query)

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # INDEXES
            # ==================================================

            indexes = [

                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                documents_user_filename_unique_idx
                ON public.documents(user_id, filename)
                """,

                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                document_chunks_document_chunk_unique_idx
                ON public.document_chunks(document_id, chunk_id)
                """,

                """
                CREATE INDEX IF NOT EXISTS
                idx_document_chunks_document
                ON public.document_chunks(document_id)
                """,

                """
                CREATE INDEX IF NOT EXISTS
                idx_conversations_user
                ON public.conversations(user_id)
                """,

                """
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation
                ON public.messages(conversation_id)
                """,
            ]

            for query in indexes:

                try:

                    cursor.execute(query)

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # FULL TEXT INDEX
            # ==================================================

            try:

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_document_chunks_content
                    ON public.document_chunks
                    USING gin (
                        to_tsvector(
                            'english',
                            COALESCE(content, '')
                        )
                    )
                    """
                )

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # HNSW INDEX
            # ==================================================

            if PGVECTOR_ENABLED:

                try:

                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS
                        idx_document_chunks_embedding_hnsw
                        ON public.document_chunks
                        USING hnsw (
                            embedding vector_cosine_ops
                        )
                        """
                    )

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # CLEAN ORPHANS
            # ==================================================

            cleanup_queries = [

                """
                DELETE FROM public.documents d
                WHERE d.user_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.users u
                    WHERE u.id = d.user_id
                )
                """,

                """
                DELETE FROM public.document_chunks c
                WHERE c.document_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.documents d
                    WHERE d.id = c.document_id
                )
                """,

                """
                DELETE FROM public.conversations c
                WHERE c.user_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.users u
                    WHERE u.id = c.user_id
                )
                """,

                """
                DELETE FROM public.messages m
                WHERE m.conversation_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.conversations c
                    WHERE c.id = m.conversation_id
                )
                """,
            ]

            for query in cleanup_queries:

                try:

                    cursor.execute(query)

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # ENSURE ADMIN
            # ==================================================

            try:

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(COALESCE(role, 'user')) = 'admin'
                    """
                )

                admin_count = cursor.fetchone()[0]

                if admin_count == 0:

                    if ADMIN_USERNAME:

                        cursor.execute(
                            """
                            UPDATE public.users
                            SET role = 'admin'
                            WHERE LOWER(username) = LOWER(%s)
                            """,
                            (
                                ADMIN_USERNAME,
                            ),
                        )

                    else:

                        cursor.execute(
                            """
                            UPDATE public.users
                            SET role = 'admin'
                            WHERE id = (
                                SELECT id
                                FROM public.users
                                ORDER BY id
                                LIMIT 1
                            )
                            """
                        )

                    conn.commit()

            except Exception:

                conn.rollback()

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                try:
                    cursor.close()
                except Exception:
                    pass

    # ========================================================
    # CREATE USER
    # ========================================================

    def create_user(
        self,
        username,
        password,
    ):

        conn = self.connect()

        cursor = None

        try:

            username = username.strip()

            if not username:

                return None

            if not password:

                return None

            cursor = conn.cursor()

            password_hash = generate_password_hash(
                password
            )

            cursor.execute(
                """
                INSERT INTO public.users (
                    username,
                    password_hash,
                    role
                )
                VALUES (
                    %s,
                    %s,
                    'user'
                )
                RETURNING id, username, role
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

            if cursor:

                cursor.close()

    # ========================================================
    # AUTHENTICATE
    # ========================================================

    def authenticate(
        self,
        username,
        password,
    ):

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
                (
                    username.strip(),
                ),
            )

            user = cursor.fetchone()

            if not user:

                conn.rollback()

                return None

            if not user.get("password_hash"):

                conn.rollback()

                return None

            valid = check_password_hash(
                user["password_hash"],
                password,
            )

            conn.rollback()

            if not valid:

                return None

            return dict(user)

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
    # SET USER ROLE
    # ========================================================

    def set_user_role(
        self,
        user_id,
        role,
    ):

        if role not in {
            "admin",
            "user",
        }:

            raise ValueError(
                "Invalid user role."
            )

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            if role == "user":

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(COALESCE(role, 'user')) = 'admin'
                    """
                )

                admin_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT role
                    FROM public.users
                    WHERE id = %s
                    """,
                    (
                        user_id,
                    ),
                )

                current = cursor.fetchone()

                if (
                    current
                    and str(current[0]).lower() == "admin"
                    and admin_count <= 1
                ):

                    conn.rollback()

                    return (
                        False,
                        "You cannot remove the last admin.",
                    )

            cursor.execute(
                """
                UPDATE public.users
                SET role = %s
                WHERE id = %s
                """,
                (
                    role,
                    user_id,
                ),
            )

            conn.commit()

            return (
                True,
                "Role updated.",
            )

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                cursor.close()

    # ========================================================
    # DELETE USER
    # ========================================================

    def delete_user(
        self,
        user_id,
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, username, role
                FROM public.users
                WHERE id = %s
                """,
                (
                    user_id,
                ),
            )

            user = cursor.fetchone()

            if not user:

                conn.rollback()

                return (
                    False,
                    "User not found.",
                )

            if str(user[2]).lower() == "admin":

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(COALESCE(role, 'user')) = 'admin'
                    """
                )

                admin_count = cursor.fetchone()[0]

                if admin_count <= 1:

                    conn.rollback()

                    return (
                        False,
                        "You cannot delete the last admin.",
                    )

            cursor.execute(
                """
                DELETE FROM public.users
                WHERE id = %s
                """,
                (
                    user_id,
                ),
            )

            conn.commit()

            return (
                True,
                "User deleted.",
            )

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
                    user_id,
                ),
            )

            conn.commit()

            return True

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                cursor.close()

    # ========================================================
    # ADMIN STATS
    # ========================================================

    def get_admin_stats(self):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            stats = {}

            queries = {

                "users":
                    "SELECT COUNT(*) FROM public.users",

                "documents":
                    "SELECT COUNT(*) FROM public.documents",

                "chunks":
                    "SELECT COUNT(*) FROM public.document_chunks",

                "conversations":
                    "SELECT COUNT(*) FROM public.conversations",

                "messages":
                    "SELECT COUNT(*) FROM public.messages",
            }

            for key, query in queries.items():

                cursor.execute(query)

                stats[key] = cursor.fetchone()[0]

            conn.rollback()

            return stats

        finally:

            if cursor:

                cursor.close()

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

        cursor = None

        try:

            cursor = conn.cursor()

            # Delete old document with same name.
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

            cursor.execute(
                """
                INSERT INTO public.documents (
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
                    chunk_size,
                    chunk_overlap,
                    Json(metadata or {}),
                ),
            )

            document_id = cursor.fetchone()[0]

            conn.commit()

            return document_id

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                cursor.close()

    # ========================================================
    # SAVE CHUNKS
    # ========================================================

    def save_chunks(
        self,
        document_id,
        chunks,
        embedding_function=None,
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            vector_available = False

            if PGVECTOR_ENABLED:

                try:

                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = 'document_chunks'
                            AND column_name = 'embedding'
                        )
                        """
                    )

                    vector_available = bool(
                        cursor.fetchone()[0]
                    )

                except Exception:

                    conn.rollback()

            for index, chunk in enumerate(chunks):

                if not isinstance(chunk, dict):

                    chunk = {
                        "content": str(chunk)
                    }

                chunk_id = chunk.get(
                    "chunk_id",
                    index,
                )

                metadata = chunk.get(
                    "metadata",
                    {},
                )

                embedding = None

                if (
                    vector_available
                    and embedding_function
                ):

                    try:

                        embedding = embedding_function(
                            chunk.get(
                                "content",
                                "",
                            )
                        )

                        if embedding is not None:

                            embedding = [
                                float(x)
                                for x in embedding
                            ]

                            if len(embedding) != PGVECTOR_DIMENSION:

                                embedding = None

                    except Exception:

                        embedding = None

                if vector_available:

                    cursor.execute(
                        """
                        INSERT INTO public.document_chunks (
                            document_id,
                            chunk_id,
                            chunk_type,
                            content,
                            page,
                            tokens,
                            characters,
                            metadata,
                            chunk_data,
                            embedding
                        )
                        VALUES (
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
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
                            chunk_data = EXCLUDED.chunk_data,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            document_id,
                            chunk_id,
                            chunk.get("chunk_type"),
                            chunk.get("content", ""),
                            chunk.get("page"),
                            chunk.get("tokens"),
                            chunk.get("characters"),
                            Json(metadata),
                            Json(chunk),
                            embedding,
                        ),
                    )

                else:

                    cursor.execute(
                        """
                        INSERT INTO public.document_chunks (
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
                            %s,%s,%s,%s,%s,%s,%s,%s,%s
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
                            chunk_id,
                            chunk.get("chunk_type"),
                            chunk.get("content", ""),
                            chunk.get("page"),
                            chunk.get("tokens"),
                            chunk.get("characters"),
                            Json(metadata),
                            Json(chunk),
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
    # GET DOCUMENTS
    # ========================================================

    def get_documents(
        self,
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
                        u.username
                    FROM public.documents d
                    LEFT JOIN public.users u
                        ON u.id = d.user_id
                    ORDER BY d.created_at DESC
                    """
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        filename,
                        file_type,
                        file_size,
                        chunking_method,
                        chunk_size,
                        chunk_overlap,
                        metadata,
                        created_at
                    FROM public.documents
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (
                        user_id,
                    ),
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
                    (
                        document_id,
                    ),
                )

            else:

                cursor.execute(
                    """
                    DELETE FROM public.documents
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        document_id,
                        user_id,
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
        retrieval_method="PostgreSQL Full-Text",
        query_embedding=None,
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            conditions = [
                "d.user_id = %s"
            ]

            base_params = [
                user_id
            ]

            if selected_document_ids:

                conditions.append(
                    "d.id = ANY(%s)"
                )

                base_params.append(
                    selected_document_ids
                )

            if chunk_types:

                conditions.append(
                    "c.chunk_type = ANY(%s)"
                )

                base_params.append(
                    chunk_types
                )

            where_sql = " AND ".join(
                conditions
            )

            if (
                retrieval_method
                == "pgvector Semantic"
                and query_embedding is not None
            ):

                query_sql = f"""
                    SELECT
                        c.*,
                        d.filename,
                        d.chunking_method,

                        1 - (
                            c.embedding <=> %s
                        ) AS similarity_score,

                        'pgvector_semantic'
                            AS search_type,

                        CASE
                            WHEN c.chunk_type = 'Table'
                            THEN 'table'
                            ELSE 'document'
                        END AS source_type

                    FROM public.document_chunks c

                    JOIN public.documents d
                        ON d.id = c.document_id

                    WHERE
                        {where_sql}
                        AND c.embedding IS NOT NULL

                    ORDER BY c.embedding <=> %s

                    LIMIT %s
                """

                params = (
                    base_params
                    + [
                        query_embedding,
                        query_embedding,
                        top_k,
                    ]
                )

            elif (
                retrieval_method
                == "Hybrid (pgvector + PostgreSQL Full-Text)"
                and query_embedding is not None
            ):

                query_sql = f"""
                    SELECT
                        c.*,
                        d.filename,
                        d.chunking_method,

                        (
                            0.70 * (
                                1 - (
                                    c.embedding <=> %s
                                )
                            )
                            +
                            0.30 * ts_rank(
                                to_tsvector(
                                    'english',
                                    COALESCE(c.content, '')
                                ),
                                plainto_tsquery(
                                    'english',
                                    %s
                                )
                            )
                        ) AS similarity_score,

                        'hybrid_pgvector_fulltext'
                            AS search_type,

                        CASE
                            WHEN c.chunk_type = 'Table'
                            THEN 'table'
                            ELSE 'document'
                        END AS source_type

                    FROM public.document_chunks c

                    JOIN public.documents d
                        ON d.id = c.document_id

                    WHERE
                        {where_sql}

                        AND c.embedding IS NOT NULL

                        AND (
                            to_tsvector(
                                'english',
                                COALESCE(c.content, '')
                            )
                            @@ plainto_tsquery(
                                'english',
                                %s
                            )

                            OR c.content ILIKE %s
                        )

                    ORDER BY similarity_score DESC

                    LIMIT %s
                """

                params = (
                    base_params
                    + [
                        query_embedding,
                        query,
                        query,
                        f"%{query}%",
                        top_k,
                    ]
                )

            else:

                query_sql = f"""
                    SELECT
                        c.*,
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
                        ) AS similarity_score,

                        'postgresql_chunk_search'
                            AS search_type,

                        CASE
                            WHEN c.chunk_type = 'Table'
                            THEN 'table'
                            ELSE 'document'
                        END AS source_type

                    FROM public.document_chunks c

                    JOIN public.documents d
                        ON d.id = c.document_id

                    WHERE
                        {where_sql}

                        AND (
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
                        )

                    ORDER BY similarity_score DESC

                    LIMIT %s
                """

                params = (
                    base_params
                    + [
                        query,
                        query,
                        f"%{query}%",
                        top_k,
                    ]
                )

            cursor.execute(
                query_sql,
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
    # CREATE CONVERSATION
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
                INSERT INTO public.conversations (
                    user_id,
                    title
                )
                VALUES (
                    %s,
                    %s
                )
                RETURNING id
                """,
                (
                    user_id,
                    title,
                ),
            )

            conversation_id = cursor.fetchone()[0]

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

    def get_conversations(
        self,
        user_id,
    ):

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
                ORDER BY created_at DESC, id DESC
                """,
                (
                    user_id,
                ),
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
    # GET MESSAGES
    # ========================================================

    def get_messages(
        self,
        conversation_id,
    ):

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
                    conversation_id,
                    role,
                    content,
                    created_at
                FROM public.messages
                WHERE conversation_id = %s
                ORDER BY id ASC
                """,
                (
                    conversation_id,
                ),
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
    # SAVE MESSAGE
    # ========================================================

    def save_message(
        self,
        conversation_id,
        role,
        content,
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO public.messages (
                    conversation_id,
                    role,
                    content
                )
                VALUES (
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    conversation_id,
                    role,
                    content,
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
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE public.conversations
                SET title = %s
                WHERE id = %s
                """,
                (
                    title,
                    conversation_id,
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
    # DELETE CONVERSATION
    # ========================================================

    def delete_conversation(
        self,
        conversation_id,
        user_id,
    ):

        conn = self.connect()

        cursor = None

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM public.conversations
                WHERE id = %s
                AND user_id = %s
                """,
                (
                    conversation_id,
                    user_id,
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


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document(
    filename,
    file_bytes,
):

    extension = Path(
        filename
    ).suffix.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

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
                    }
                )

        return pages

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    if extension == ".docx":

        if DocxDocument is None:

            raise RuntimeError(
                "python-docx is not installed."
            )

        document = DocxDocument(
            io.BytesIO(file_bytes)
        )

        text_parts = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                text_parts.append(text)

        content = "\n\n".join(
            text_parts
        )

        return [
            {
                "page": None,
                "content": content,
                "chunk_type": "Document",
            }
        ]

    # --------------------------------------------------------
    # XLSX
    # --------------------------------------------------------

    if extension in {
        ".xlsx",
        ".xls",
    }:

        sheets = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=None,
        )

        result = []

        for sheet_name, dataframe in sheets.items():

            csv_text = dataframe.to_csv(
                index=False
            )

            result.append(
                {
                    "page": sheet_name,
                    "content": csv_text,
                    "chunk_type": "Table",
                    "metadata": {
                        "sheet": sheet_name,
                    },
                }
            )

        return result

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extension == ".csv":

        dataframe = pd.read_csv(
            io.BytesIO(file_bytes)
        )

        return [
            {
                "page": None,
                "content": dataframe.to_csv(
                    index=False
                ),
                "chunk_type": "Table",
            }
        ]

    # --------------------------------------------------------
    # TXT / MARKDOWN
    # --------------------------------------------------------

    if extension in {
        ".txt",
        ".md",
        ".markdown",
    }:

        content = file_bytes.decode(
            "utf-8",
            errors="replace",
        )

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

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# FALLBACK CHUNKER
# ============================================================

def fallback_chunk(
    text,
    chunk_size,
    chunk_overlap,
    chunk_type="Document",
    page=None,
    metadata=None,
):

    text = str(text or "").strip()

    if not text:

        return []

    chunk_size = max(
        100,
        int(chunk_size),
    )

    chunk_overlap = max(
        0,
        min(
            int(chunk_overlap),
            chunk_size - 1,
        ),
    )

    chunks = []

    start = 0

    index = 0

    while start < len(text):

        end = min(
            len(text),
            start + chunk_size,
        )

        content = text[
            start:end
        ].strip()

        if content:

            chunks.append(
                {
                    "chunk_id": index,
                    "chunk_type": chunk_type,
                    "content": content,
                    "page": page,
                    "tokens": max(
                        1,
                        len(
                            content.split()
                        ),
                    ),
                    "characters": len(
                        content
                    ),
                    "metadata": metadata or {},
                }
            )

            index += 1

        if end >= len(text):

            break

        start = max(
            start + 1,
            end - chunk_overlap,
        )

    return chunks


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_document(
    extracted_pages,
    method,
    chunk_size,
    chunk_overlap,
):

    all_chunks = []

    # --------------------------------------------------------
    # Try project ChunkingEngine first
    # --------------------------------------------------------

    try:

        engine = ChunkingEngine()

        methods = [
            "chunk",
            "chunk_text",
            "create_chunks",
            "process",
            "split",
        ]

        selected_method = None

        for name in methods:

            candidate = getattr(
                engine,
                name,
                None,
            )

            if callable(candidate):

                selected_method = candidate

                break

        if selected_method:

            for page_data in extracted_pages:

                content = page_data.get(
                    "content",
                    "",
                )

                try:

                    result = selected_method(
                        content,
                        method=method,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                except TypeError:

                    try:

                        result = selected_method(
                            content,
                            chunk_size,
                            chunk_overlap,
                        )

                    except Exception:

                        result = None

                if result:

                    if isinstance(
                        result,
                        list,
                    ):

                        for item in result:

                            if isinstance(
                                item,
                                dict,
                            ):

                                chunk = dict(item)

                            else:

                                chunk = {
                                    "content": str(item)
                                }

                            chunk.setdefault(
                                "chunk_type",
                                page_data.get(
                                    "chunk_type",
                                    "Document",
                                ),
                            )

                            chunk.setdefault(
                                "page",
                                page_data.get(
                                    "page"
                                ),
                            )

                            chunk.setdefault(
                                "characters",
                                len(
                                    chunk.get(
                                        "content",
                                        "",
                                    )
                                ),
                            )

                            chunk.setdefault(
                                "tokens",
                                max(
                                    1,
                                    len(
                                        chunk.get(
                                            "content",
                                            "",
                                        ).split()
                                    ),
                                ),
                            )

                            all_chunks.append(
                                chunk
                            )

                    continue

                fallback = fallback_chunk(
                    content,
                    chunk_size,
                    chunk_overlap,
                    page_data.get(
                        "chunk_type",
                        "Document",
                    ),
                    page_data.get(
                        "page"
                    ),
                    page_data.get(
                        "metadata",
                        {},
                    ),
                )

                all_chunks.extend(
                    fallback
                )

            if all_chunks:

                for i, chunk in enumerate(
                    all_chunks
                ):

                    chunk["chunk_id"] = i

                return all_chunks

    except Exception:

        pass

    # --------------------------------------------------------
    # Fallback implementation
    # --------------------------------------------------------

    for page_data in extracted_pages:

        chunks = fallback_chunk(
            page_data.get(
                "content",
                "",
            ),
            chunk_size,
            chunk_overlap,
            page_data.get(
                "chunk_type",
                "Document",
            ),
            page_data.get(
                "page",
            ),
            page_data.get(
                "metadata",
                {},
            ),
        )

        all_chunks.extend(
            chunks
        )

    for i, chunk in enumerate(
        all_chunks
    ):

        chunk["chunk_id"] = i

    return all_chunks


# ============================================================
# SAFE GEMINI MODEL CREATION
# ============================================================

@st.cache_resource
def create_llm():

    try:

        return GeminiModel()

    except TypeError:

        try:

            return GeminiModel(
                model_name=os.getenv(
                    "GEMINI_MODEL",
                    "gemini-2.5-flash",
                )
            )

        except Exception:

            return None

    except Exception:

        return None


# ============================================================
# OPTIONAL EMBEDDING FUNCTION
# ============================================================

def create_embedding_function():

    llm = st.session_state.get(
        "llm"
    )

    if llm is None:

        return None

    possible_methods = [
        "embed",
        "embed_text",
        "get_embedding",
        "create_embedding",
    ]

    for method_name in possible_methods:

        method = getattr(
            llm,
            method_name,
            None,
        )

        if not callable(method):

            continue

        def embedding_function(
            text,
            method=method,
        ):

            try:

                result = method(
                    text
                )

                if isinstance(
                    result,
                    dict,
                ):

                    result = (
                        result.get(
                            "embedding"
                        )
                        or result.get(
                            "values"
                        )
                    )

                return result

            except Exception:

                return None

        return embedding_function

    return None


# ============================================================
# INITIALIZE DATABASE
# ============================================================

@st.cache_resource
def initialize_database():

    store = PostgreSQLStore()

    store.initialize()

    return store


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {

        "authenticated":
            False,

        "user":
            None,

        "db":
            None,

        "llm":
            None,

        "conversation_id":
            None,

        "messages":
            [],

        "selected_document_ids":
            [],

        "selected_chunk_types":
            [],

        "top_k":
            DEFAULT_TOP_K,

        "retrieval_method":
            "PostgreSQL Full-Text",

        "login_mode":
            "Login",

        "admin_page":
            "Dashboard",
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# ============================================================
# DATABASE CONNECTION
# ============================================================

if st.session_state.db is None:

    try:

        st.session_state.db = initialize_database()

    except Exception as exc:

        st.error(
            f"Database connection failed: {exc}"
        )

        st.info(
            "Check DATABASE_URL or PG_HOST, "
            "PG_PORT, PG_DATABASE, PG_USER and "
            "PG_PASSWORD."
        )

        st.stop()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title(
        "📄 Enterprise Document Intelligence"
    )

    st.caption(
        "PostgreSQL + RAG + Gemini"
    )

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register",
        ]
    )

    with login_tab:

        with st.form(
            "login_form"
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

                            st.session_state.llm = create_llm()

                            st.rerun()

                        else:

                            st.error(
                                "Invalid username or password."
                            )

                    except Exception as exc:

                        st.error(
                            f"Login failed: {exc}"
                        )

    with register_tab:

        with st.form(
            "register_form"
        ):

            new_username = st.text_input(
                "New username"
            )

            new_password = st.text_input(
                "New password",
                type="password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )

            if submitted:

                if not new_username:

                    st.error(
                        "Username is required."
                    )

                elif not new_password:

                    st.error(
                        "Password is required."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                elif len(new_password) < 6:

                    st.error(
                        "Password must contain at least 6 characters."
                    )

                else:

                    result = (
                        st.session_state.db.create_user(
                            new_username,
                            new_password,
                        )
                    )

                    if result:

                        st.success(
                            "Account created. You can now log in."
                        )

                    else:

                        st.error(
                            "Could not create account. "
                            "The username may already exist."
                        )


# ============================================================
# STOP HERE IF NOT LOGGED IN
# ============================================================

if not st.session_state.authenticated:

    login_page()

    st.stop()