import io
import os
import json
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
                "psycopg2 is not installed. " "Add psycopg2-binary to requirements.txt."
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

                register_vector(self.connection)

            except Exception:

                pass

        cursor = self.connection.cursor()

        try:

            cursor.execute("SET search_path TO public")

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
            (table_name,),
        )

        return bool(cursor.fetchone()[0])

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

        return bool(cursor.fetchone()[0])

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

        query = sql.SQL("""
            ALTER TABLE public.{}
            ADD COLUMN IF NOT EXISTS {} {}
            """).format(
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

        query = sql.SQL("""
            ALTER TABLE public.{}
            DROP CONSTRAINT IF EXISTS {}
            """).format(
            sql.Identifier(table_name),
            sql.Identifier(constraint_name),
        )

        cursor.execute(query)

    # ========================================================
    # INITIALIZE / MIGRATE DATABASE
    # ========================================================

    def initialize(self):

        conn = self.connect()
        cursor = None

        try:

            conn.rollback()

            cursor = conn.cursor()

            cursor.execute("SET search_path TO public")

            # ==================================================
            # USERS
            # ==================================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255),
                    password_hash TEXT,
                    role VARCHAR(50)
                        DEFAULT 'user',
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

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

            cursor.execute("""
                UPDATE public.users
                SET role = 'user'
                WHERE role IS NULL
                   OR LOWER(role)
                      NOT IN ('admin', 'user')
                """)

            self.add_column_if_missing(
                cursor,
                "users",
                "created_at",
                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            )

            conn.commit()

            try:

                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    users_username_unique_idx
                    ON public.users(username)
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # DOCUMENTS
            # ==================================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                public.documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    filename TEXT,
                    file_type TEXT,
                    file_size BIGINT,
                    file_data BYTEA,
                    chunking_method TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER,
                    metadata JSONB
                        DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            conn.commit()

            document_columns = {
                "user_id": "INTEGER",
                "filename": "TEXT",
                "file_type": "TEXT",
                "file_size": "BIGINT",
                "file_data": "BYTEA",
                "chunking_method": "TEXT",
                "chunk_size": "INTEGER",
                "chunk_overlap": "INTEGER",
                "metadata": "JSONB DEFAULT '{}'::jsonb",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for (
                column_name,
                definition,
            ) in document_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "documents",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # PGVECTOR / SEMANTIC RETRIEVAL
            # ==================================================

            if PGVECTOR_ENABLED:

                try:

                    cursor.execute("""
                        CREATE EXTENSION IF NOT EXISTS vector
                        """)

                    conn.commit()

                    self.add_column_if_missing(
                        cursor,
                        "document_chunks",
                        "embedding",
                        f"vector({PGVECTOR_DIMENSION})",
                    )

                    conn.commit()

                    try:

                        cursor.execute("""
                            CREATE INDEX IF NOT EXISTS
                            idx_document_chunks_embedding_hnsw
                            ON public.document_chunks
                            USING hnsw (
                                embedding
                                vector_cosine_ops
                            )
                            """)

                        conn.commit()

                    except Exception:

                        conn.rollback()

                except Exception:

                    conn.rollback()

            # ==================================================
            # DOCUMENT CHUNKS
            # ==================================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                public.document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER,
                    chunk_id INTEGER,
                    chunk_type TEXT,
                    content TEXT,
                    page INTEGER,
                    tokens INTEGER,
                    characters INTEGER,
                    metadata JSONB
                        DEFAULT '{}'::jsonb,
                    chunk_data JSONB
                        DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            conn.commit()

            chunk_columns = {
                "document_id": "INTEGER",
                "chunk_id": "INTEGER",
                "chunk_type": "TEXT",
                "content": "TEXT",
                "page": "INTEGER",
                "tokens": "INTEGER",
                "characters": "INTEGER",
                "metadata": "JSONB DEFAULT '{}'::jsonb",
                "chunk_data": "JSONB DEFAULT '{}'::jsonb",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for (
                column_name,
                definition,
            ) in chunk_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "document_chunks",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # CONVERSATIONS
            # ==================================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                public.conversations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            conn.commit()

            conversation_columns = {
                "user_id": "INTEGER",
                "title": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for (
                column_name,
                definition,
            ) in conversation_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "conversations",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # MESSAGES
            # ==================================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                public.messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            conn.commit()

            message_columns = {
                "conversation_id": "INTEGER",
                "role": "TEXT",
                "content": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }

            for (
                column_name,
                definition,
            ) in message_columns.items():

                self.add_column_if_missing(
                    cursor,
                    "messages",
                    column_name,
                    definition,
                )

                conn.commit()

            # ==================================================
            # REMOVE OLD CONSTRAINTS
            # ==================================================

            for (
                table_name,
                constraint_name,
            ) in [
                (
                    "documents",
                    "documents_user_id_fkey",
                ),
                (
                    "document_chunks",
                    "document_chunks_document_id_fkey",
                ),
                (
                    "conversations",
                    "conversations_user_id_fkey",
                ),
                (
                    "messages",
                    "messages_conversation_id_fkey",
                ),
            ]:

                try:

                    self.remove_constraint_if_exists(
                        cursor,
                        table_name,
                        constraint_name,
                    )

                    conn.commit()

                except Exception:

                    conn.rollback()

            # ==================================================
            # CLEAN ORPHAN DOCUMENTS
            # ==================================================

            try:

                cursor.execute("""
                    DELETE FROM public.documents d
                    WHERE d.user_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1
                        FROM public.users u
                        WHERE u.id = d.user_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # CLEAN ORPHAN CHUNKS
            # ==================================================

            try:

                cursor.execute("""
                    DELETE FROM public.document_chunks c
                    WHERE c.document_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1
                        FROM public.documents d
                        WHERE d.id = c.document_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # CLEAN ORPHAN CONVERSATIONS
            # ==================================================

            try:

                cursor.execute("""
                    DELETE FROM public.conversations c
                    WHERE c.user_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1
                        FROM public.users u
                        WHERE u.id = c.user_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # CLEAN ORPHAN MESSAGES
            # ==================================================

            try:

                cursor.execute("""
                    DELETE FROM public.messages m
                    WHERE m.conversation_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1
                        FROM public.conversations c
                        WHERE c.id = m.conversation_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # FOREIGN KEYS
            # ==================================================

            try:

                cursor.execute("""
                    ALTER TABLE public.documents
                    ADD CONSTRAINT
                    documents_user_id_fkey
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    ALTER TABLE public.document_chunks
                    ADD CONSTRAINT
                    document_chunks_document_id_fkey
                    FOREIGN KEY (document_id)
                    REFERENCES public.documents(id)
                    ON DELETE CASCADE
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    ALTER TABLE public.conversations
                    ADD CONSTRAINT
                    conversations_user_id_fkey
                    FOREIGN KEY (user_id)
                    REFERENCES public.users(id)
                    ON DELETE CASCADE
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    ALTER TABLE public.messages
                    ADD CONSTRAINT
                    messages_conversation_id_fkey
                    FOREIGN KEY (conversation_id)
                    REFERENCES public.conversations(id)
                    ON DELETE CASCADE
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # INDEXES
            # ==================================================

            try:

                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    documents_user_filename_unique_idx
                    ON public.documents(
                        user_id,
                        filename
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    document_chunks_document_chunk_unique_idx
                    ON public.document_chunks(
                        document_id,
                        chunk_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_document_chunks_document
                    ON public.document_chunks(
                        document_id
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            try:

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_document_chunks_content
                    ON public.document_chunks
                    USING gin (
                        to_tsvector(
                            'english',
                            COALESCE(
                                content,
                                ''
                            )
                        )
                    )
                    """)

                conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # ENSURE THERE IS AN ADMIN
            # ==================================================

            try:

                cursor.execute("""
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(
                        COALESCE(role, 'user')
                    ) = 'admin'
                    """)

                admin_count = cursor.fetchone()[0]

                if admin_count == 0:

                    if ADMIN_USERNAME:

                        cursor.execute(
                            """
                            UPDATE public.users
                            SET role = 'admin'
                            WHERE LOWER(username)
                                = LOWER(%s)
                            """,
                            (ADMIN_USERNAME,),
                        )

                    else:

                        cursor.execute("""
                            UPDATE public.users
                            SET role = 'admin'
                            WHERE id = (
                                SELECT id
                                FROM public.users
                                ORDER BY id
                                LIMIT 1
                            )
                            """)

                    conn.commit()

            except Exception:

                conn.rollback()

            # ==================================================
            # FINAL SCHEMA VERIFICATION
            # ==================================================

            required_columns = {
                "users": [
                    "id",
                    "username",
                    "password_hash",
                    "role",
                    "created_at",
                ],
                "documents": [
                    "id",
                    "user_id",
                    "filename",
                    "file_type",
                    "file_size",
                    "file_data",
                    "chunking_method",
                    "chunk_size",
                    "chunk_overlap",
                    "metadata",
                    "created_at",
                ],
                "document_chunks": [
                    "id",
                    "document_id",
                    "chunk_id",
                    "chunk_type",
                    "content",
                    "page",
                    "tokens",
                    "characters",
                    "metadata",
                    "chunk_data",
                    "created_at",
                ],
                "conversations": [
                    "id",
                    "user_id",
                    "title",
                    "created_at",
                ],
                "messages": [
                    "id",
                    "conversation_id",
                    "role",
                    "content",
                    "created_at",
                ],
            }

            if PGVECTOR_ENABLED:

                required_columns["document_chunks"].append("embedding")

            missing = []

            for (
                table_name,
                columns,
            ) in required_columns.items():

                for column_name in columns:

                    if not self.column_exists(
                        cursor,
                        table_name,
                        column_name,
                    ):

                        missing.append(f"{table_name}.{column_name}")

            if missing:

                raise RuntimeError(
                    "PostgreSQL migration failed. " "Missing columns: " + ", ".join(missing)
                )

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

            cursor = conn.cursor()

            password_hash = generate_password_hash(password)

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
                    username.strip(),
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

                try:
                    cursor.close()
                except Exception:
                    pass

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

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    role
                FROM public.users
                WHERE username = %s
                """,
                (username.strip(),),
            )

            user = cursor.fetchone()

            if not user:

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
    # ADMIN USER MANAGEMENT
    # ========================================================

    def get_users(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT DISTINCT ON (id)
                    id,
                    username,
                    role,
                    created_at
                FROM public.users
                ORDER BY id, created_at DESC
                """)

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                try:
                    cursor.close()
                except Exception:
                    pass

    def set_user_role(
        self,
        user_id,
        role,
    ):

        if role not in {
            "admin",
            "user",
        }:

            raise ValueError("Invalid user role.")

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            if role == "user":

                cursor.execute("""
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(
                        COALESCE(role, 'user')
                    ) = 'admin'
                    """)

                admin_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT role
                    FROM public.users
                    WHERE id = %s
                    """,
                    (user_id,),
                )

                current = cursor.fetchone()

                if current and str(current[0]).lower() == "admin" and admin_count <= 1:

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

                try:
                    cursor.close()
                except Exception:
                    pass

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
                SELECT
                    id,
                    username,
                    role
                FROM public.users
                WHERE id = %s
                """,
                (user_id,),
            )

            user = cursor.fetchone()

            if not user:

                conn.rollback()

                return (
                    False,
                    "User not found.",
                )

            if str(user[2]).lower() == "admin":

                cursor.execute("""
                    SELECT COUNT(*)
                    FROM public.users
                    WHERE LOWER(
                        COALESCE(role, 'user')
                    ) = 'admin'
                    """)

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
                (user_id,),
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

                try:
                    cursor.close()
                except Exception:
                    pass

    def reset_user_password(
        self,
        user_id,
        password,
    ):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            password_hash = generate_password_hash(password)

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

                try:
                    cursor.close()
                except Exception:
                    pass

    def get_admin_stats(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor()

            stats = {}

            queries = {
                "users": "SELECT COUNT(*) FROM public.users",
                "documents": "SELECT COUNT(*) FROM public.documents",
                "chunks": "SELECT COUNT(*) FROM public.document_chunks",
                "conversations": "SELECT COUNT(*) FROM public.conversations",
                "messages": "SELECT COUNT(*) FROM public.messages",
            }

            for (
                key,
                query,
            ) in queries.items():

                cursor.execute(query)

                stats[key] = cursor.fetchone()[0]

            conn.rollback()

            return stats

        except Exception:

            conn.rollback()

            raise

        finally:

            if cursor:

                try:
                    cursor.close()
                except Exception:
                    pass

    def get_all_documents_admin(self):

        conn = self.connect()
        cursor = None

        try:

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    d.id,
                    d.user_id,
                    u.username,
                    d.filename,
                    d.file_type,
                    d.file_size,
                    d.chunking_method,
                    d.chunk_size,
                    d.chunk_overlap,
                    d.metadata,
                    d.created_at
                FROM public.documents d
                LEFT JOIN public.users u
                    ON u.id = d.user_id
                ORDER BY d.created_at DESC
                """)

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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

            # Self-heal legacy databases.

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunking_method",
                "TEXT",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunk_size",
                "INTEGER",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunk_overlap",
                "INTEGER",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "metadata",
                "JSONB DEFAULT '{}'::jsonb",
            )

            conn.commit()

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

                try:
                    cursor.close()
                except Exception:
                    pass

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

                    self.add_column_if_missing(
                        cursor,
                        "document_chunks",
                        "embedding",
                        f"vector({PGVECTOR_DIMENSION})",
                    )

                    conn.commit()

                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = 'document_chunks'
                            AND column_name = 'embedding'
                        )
                        """)

                    vector_available = bool(cursor.fetchone()[0])

                except Exception:

                    conn.rollback()

            for index, chunk in enumerate(chunks):

                if not isinstance(
                    chunk,
                    dict,
                ):

                    chunk = {"content": str(chunk)}

                chunk_id = chunk.get(
                    "chunk_id",
                    index,
                )

                metadata = chunk.get(
                    "metadata",
                    {},
                )

                embedding = None

                if vector_available and embedding_function:

                    content = chunk.get(
                        "content",
                        "",
                    )

                    try:

                        embedding = embedding_function(content)

                        if embedding is not None:

                            embedding = [float(x) for x in embedding]

                            if len(embedding) != PGVECTOR_DIMENSION:

                                embedding = None

                    except Exception:

                        embedding = None

                if vector_available:

                    cursor.execute(
                        """
                        INSERT INTO
                        public.document_chunks (
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
                            %s,
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
                        ON CONFLICT (
                            document_id,
                            chunk_id
                        )
                        DO UPDATE SET
                            chunk_type =
                                EXCLUDED.chunk_type,
                            content =
                                EXCLUDED.content,
                            page =
                                EXCLUDED.page,
                            tokens =
                                EXCLUDED.tokens,
                            characters =
                                EXCLUDED.characters,
                            metadata =
                                EXCLUDED.metadata,
                            chunk_data =
                                EXCLUDED.chunk_data,
                            embedding =
                                EXCLUDED.embedding
                        """,
                        (
                            document_id,
                            chunk_id,
                            chunk.get("chunk_type"),
                            chunk.get(
                                "content",
                                "",
                            ),
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
                        INSERT INTO
                        public.document_chunks (
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
                        ON CONFLICT (
                            document_id,
                            chunk_id
                        )
                        DO UPDATE SET
                            chunk_type =
                                EXCLUDED.chunk_type,
                            content =
                                EXCLUDED.content,
                            page =
                                EXCLUDED.page,
                            tokens =
                                EXCLUDED.tokens,
                            characters =
                                EXCLUDED.characters,
                            metadata =
                                EXCLUDED.metadata,
                            chunk_data =
                                EXCLUDED.chunk_data
                        """,
                        (
                            document_id,
                            chunk_id,
                            chunk.get("chunk_type"),
                            chunk.get(
                                "content",
                                "",
                            ),
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

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunking_method",
                "TEXT",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunk_size",
                "INTEGER",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "chunk_overlap",
                "INTEGER",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "metadata",
                "JSONB DEFAULT '{}'::jsonb",
            )

            self.add_column_if_missing(
                cursor,
                "documents",
                "created_at",
                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            )

            conn.commit()

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
                WHERE (
                    %s = TRUE
                    OR user_id = %s
                )
                ORDER BY created_at DESC
                """,
                (
                    is_admin,
                    user_id,
                ),
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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
    # SEARCH DOCUMENT CHUNKS
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

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            conditions = ["d.user_id = %s"]

            base_params = [user_id]

            if selected_document_ids:

                conditions.append("d.id = ANY(%s)")

                base_params.append(selected_document_ids)

            if chunk_types:

                conditions.append("c.chunk_type = ANY(%s)")

                base_params.append(chunk_types)

            where_sql = " AND ".join(conditions)

            if retrieval_method == "pgvector Semantic" and query_embedding is not None:

                q = f"""
                    SELECT
                        c.*,
                        d.filename,
                        d.chunking_method,

                        1 - (
                            c.embedding
                            <=> %s
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

                    ORDER BY
                        c.embedding <=> %s

                    LIMIT %s
                """

                params = base_params + [
                    query_embedding,
                    query_embedding,
                    top_k,
                ]

            elif (
                retrieval_method == "Hybrid (pgvector + PostgreSQL Full-Text)"
                and query_embedding is not None
            ):

                q = f"""
                    SELECT
                        c.*,
                        d.filename,
                        d.chunking_method,

                        (
                            0.70 * (
                                1 - (
                                    c.embedding
                                    <=> %s
                                )
                            )
                            +
                            0.30 * ts_rank(
                                to_tsvector(
                                    'english',
                                    COALESCE(
                                        c.content,
                                        ''
                                    )
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
                                COALESCE(
                                    c.content,
                                    ''
                                )
                            )
                            @@ plainto_tsquery(
                                'english',
                                %s
                            )

                            OR c.content ILIKE %s
                        )

                    ORDER BY
                        similarity_score DESC

                    LIMIT %s
                """

                params = base_params + [
                    query_embedding,
                    query,
                    query,
                    f"%{query}%",
                    top_k,
                ]

            else:

                q = f"""
                    SELECT
                        c.*,
                        d.filename,
                        d.chunking_method,

                        ts_rank(
                            to_tsvector(
                                'english',
                                COALESCE(
                                    c.content,
                                    ''
                                )
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
                                COALESCE(
                                    c.content,
                                    ''
                                )
                            )
                            @@ plainto_tsquery(
                                'english',
                                %s
                            )

                            OR LOWER(
                                COALESCE(
                                    c.content,
                                    ''
                                )
                            )
                            LIKE LOWER(%s)
                        )

                    ORDER BY
                        similarity_score DESC

                    LIMIT %s
                """

                params = base_params + [
                    query,
                    query,
                    f"%{query}%",
                    top_k,
                ]

            cursor.execute(
                q,
                params,
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                public.conversations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            conn.commit()

            self.add_column_if_missing(
                cursor,
                "conversations",
                "user_id",
                "INTEGER",
            )

            self.add_column_if_missing(
                cursor,
                "conversations",
                "title",
                "TEXT",
            )

            self.add_column_if_missing(
                cursor,
                "conversations",
                "created_at",
                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            )

            conn.commit()

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

                try:
                    cursor.close()
                except Exception:
                    pass

    # ========================================================
    # GET CONVERSATIONS
    # ========================================================

    def get_conversations(self, user_id):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

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
                (user_id,),
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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
    # GET CONVERSATION
    # ========================================================

    def get_conversation(
        self,
        conversation_id,
        user_id,
    ):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT
                    id,
                    user_id,
                    title,
                    created_at
                FROM public.conversations
                WHERE id = %s
                AND user_id = %s
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

            conn.rollback()

            if row is None:
                return None

            return dict(row)

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
    # UPDATE CONVERSATION TITLE
    # ========================================================

    def update_conversation_title(
        self,
        conversation_id,
        user_id,
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
                AND user_id = %s
                """,
                (
                    title,
                    conversation_id,
                    user_id,
                ),
            )

            updated = cursor.rowcount > 0

            conn.commit()

            return updated

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
                try:
                    cursor.close()
                except Exception:
                    pass

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
                RETURNING id
                """,
                (
                    conversation_id,
                    role,
                    content,
                ),
            )

            message_id = cursor.fetchone()[0]

            conn.commit()

            return message_id

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
    # GET MESSAGES
    # ========================================================

    def get_messages(
        self,
        conversation_id,
        user_id,
    ):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT
                    m.id,
                    m.conversation_id,
                    m.role,
                    m.content,
                    m.created_at
                FROM public.messages m
                JOIN public.conversations c
                    ON c.id = m.conversation_id
                WHERE m.conversation_id = %s
                AND c.user_id = %s
                ORDER BY m.created_at ASC, m.id ASC
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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
    # GET DOCUMENT
    # ========================================================

    def get_document(
        self,
        document_id,
        user_id=None,
        is_admin=False,
    ):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if is_admin:
                cursor.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        filename,
                        file_type,
                        file_size,
                        file_data,
                        chunking_method,
                        chunk_size,
                        chunk_overlap,
                        metadata,
                        created_at
                    FROM public.documents
                    WHERE id = %s
                    """,
                    (document_id,),
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
                        file_data,
                        chunking_method,
                        chunk_size,
                        chunk_overlap,
                        metadata,
                        created_at
                    FROM public.documents
                    WHERE id = %s
                    AND user_id = %s
                    """,
                    (
                        document_id,
                        user_id,
                    ),
                )

            row = cursor.fetchone()

            conn.rollback()

            if row is None:
                return None

            return dict(row)

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
    # GET DOCUMENT CHUNKS
    # ========================================================

    def get_document_chunks(
        self,
        document_id,
        user_id=None,
        is_admin=False,
    ):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if is_admin:
                cursor.execute(
                    """
                    SELECT
                        c.*
                    FROM public.document_chunks c
                    JOIN public.documents d
                        ON d.id = c.document_id
                    WHERE c.document_id = %s
                    ORDER BY c.chunk_id
                    """,
                    (document_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        c.*
                    FROM public.document_chunks c
                    JOIN public.documents d
                        ON d.id = c.document_id
                    WHERE c.document_id = %s
                    AND d.user_id = %s
                    ORDER BY c.chunk_id
                    """,
                    (
                        document_id,
                        user_id,
                    ),
                )

            rows = cursor.fetchall()

            conn.rollback()

            return [dict(row) for row in rows]

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
                    (document_id,),
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
                try:
                    cursor.close()
                except Exception:
                    pass

    # ========================================================
    # GET USER DOCUMENT IDS
    # ========================================================

    def get_user_document_ids(
        self,
        user_id,
    ):
        conn = self.connect()
        cursor = None

        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id
                FROM public.documents
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            rows = cursor.fetchall()

            conn.rollback()

            return [row[0] for row in rows]

        except Exception:
            conn.rollback()
            raise

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass


# ============================================================
# DATABASE INSTANCE
# ============================================================


@st.cache_resource
def get_database():
    database = PostgreSQLStore()
    database.initialize()
    return database


# ============================================================
# MODEL INSTANCE
# ============================================================


@st.cache_resource
def get_model():
    return GeminiModel()


# ============================================================
# CHUNKING ENGINE
# ============================================================


@st.cache_resource
def get_chunking_engine():
    return ChunkingEngine()


# ============================================================
# SAFE MODEL HELPERS
# ============================================================


def call_model(
    model,
    prompt,
):
    """
    Supports several common GeminiModel interfaces so the
    application can continue working if model.py exposes
    generate(), generate_text(), invoke(), or ask().
    """

    methods = [
        "generate",
        "generate_text",
        "invoke",
        "ask",
    ]

    for method_name in methods:
        method = getattr(
            model,
            method_name,
            None,
        )

        if callable(method):
            result = method(prompt)

            if isinstance(
                result,
                str,
            ):
                return result

            if hasattr(
                result,
                "text",
            ):
                return str(result.text)

            if isinstance(
                result,
                dict,
            ):
                for key in (
                    "text",
                    "response",
                    "content",
                    "answer",
                ):
                    if key in result:
                        return str(result[key])

            return str(result)

    raise RuntimeError("GeminiModel does not expose a supported " "generation method.")


# ============================================================
# EMBEDDING HELPER
# ============================================================


def create_embedding(
    model,
    text,
):
    """
    Attempts to obtain an embedding from GeminiModel.

    If the model implementation does not expose an embedding
    method, semantic retrieval will gracefully fall back to
    PostgreSQL full-text retrieval.
    """

    methods = [
        "embed",
        "embedding",
        "embed_text",
        "create_embedding",
        "get_embedding",
    ]

    for method_name in methods:
        method = getattr(
            model,
            method_name,
            None,
        )

        if callable(method):
            try:
                result = method(
                    text,
                    model=EMBEDDING_MODEL,
                )
            except TypeError:
                try:
                    result = method(text)
                except Exception:
                    continue
            except Exception:
                continue

            if isinstance(
                result,
                dict,
            ):
                for key in (
                    "embedding",
                    "values",
                    "vector",
                ):
                    if key in result:
                        result = result[key]
                        break

            try:
                values = [float(x) for x in result]
            except Exception:
                continue

            if len(values) == PGVECTOR_DIMENSION:
                return values

    return None


# ============================================================
# CHUNKING HELPER
# ============================================================


def run_chunking(
    engine,
    file_bytes,
    filename,
    chunking_method,
    chunk_size,
    chunk_overlap,
):
    """
    Attempts to use the ChunkingEngine implementation supplied
    by chunking.py.

    The function supports common method names while preserving
    the application's existing database format.
    """

    methods = [
        "chunk_file",
        "process_file",
        "create_chunks",
        "chunk",
        "process",
    ]

    file_object = io.BytesIO(file_bytes)

    for method_name in methods:
        method = getattr(
            engine,
            method_name,
            None,
        )

        if not callable(method):
            continue

        attempts = [
            {
                "file": file_object,
                "filename": filename,
                "method": chunking_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            {
                "file": file_object,
                "filename": filename,
                "chunking_method": chunking_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            {
                "file_bytes": file_bytes,
                "filename": filename,
                "chunking_method": chunking_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        ]

        for kwargs in attempts:
            try:
                result = method(**kwargs)

                if result is not None:
                    return result

            except TypeError:
                continue

            except Exception:
                raise

        file_object.seek(0)

    raise RuntimeError(
        "ChunkingEngine does not expose a supported " "chunking method. Check chunking.py."
    )


# ============================================================
# NORMALIZE CHUNKS
# ============================================================


def normalize_chunks(
    chunks,
):
    if chunks is None:
        return []

    if isinstance(
        chunks,
        dict,
    ):
        for key in (
            "chunks",
            "data",
            "results",
        ):
            if key in chunks:
                chunks = chunks[key]
                break

    if not isinstance(
        chunks,
        (list, tuple),
    ):
        chunks = [chunks]

    normalized = []

    for index, chunk in enumerate(chunks):
        if isinstance(
            chunk,
            dict,
        ):
            item = dict(chunk)

            if "content" not in item:
                for key in (
                    "text",
                    "page_content",
                    "chunk",
                ):
                    if key in item:
                        item["content"] = str(item[key])
                        break

            item.setdefault(
                "content",
                "",
            )

        else:
            item = {"content": str(chunk)}

        item.setdefault(
            "chunk_id",
            index,
        )

        content = str(
            item.get(
                "content",
                "",
            )
        )

        item["content"] = content

        item.setdefault(
            "characters",
            len(content),
        )

        if "tokens" not in item:
            item["tokens"] = max(
                1,
                len(content.split()),
            )

        item.setdefault(
            "chunk_type",
            "Text",
        )

        item.setdefault(
            "metadata",
            {},
        )

        normalized.append(item)

    return normalized


# ============================================================
# FILE TYPE HELPER
# ============================================================


def get_file_type(
    filename,
):
    suffix = Path(filename).suffix.lower()

    mapping = {
        ".pdf": "PDF",
        ".docx": "DOCX",
        ".doc": "DOC",
        ".txt": "TXT",
        ".md": "Markdown",
        ".csv": "CSV",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".json": "JSON",
        ".html": "HTML",
        ".htm": "HTML",
    }

    return mapping.get(
        suffix,
        suffix.replace(
            ".",
            "",
        ).upper()
        or "Unknown",
    )


# ============================================================
# SESSION STATE
# ============================================================


def initialize_session_state():
    defaults = {
        "authenticated": False,
        "user": None,
        "conversation_id": None,
        "selected_document_ids": [],
        "messages": [],
        "login_error": None,
        "page": "Chat",
        "auth_view": "login",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# DATABASE CONNECTION
# ============================================================

try:
    db = get_database()
except Exception as exc:
    st.error("Unable to initialize PostgreSQL.")
    st.exception(exc)
    st.stop()


# ============================================================
# LOGIN
# ============================================================


def login_screen():
    st.title("📄 Enterprise Document Intelligence")

    st.caption("Secure document ingestion, retrieval, " "and AI-powered question answering.")

    st.divider()

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.subheader("Sign in")

        with st.form(
            "login_form",
            clear_on_submit=False,
        ):
            username = st.text_input(
                "Username",
                autocomplete="username",
                key="login_username",
            )

            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
                key="login_password",
            )

            submitted = st.form_submit_button(
                "Sign in",
                use_container_width=True,
            )

        if submitted:
            if not username.strip():
                st.error("Enter your username.")
                return

            if not password:
                st.error("Enter your password.")
                return

            try:
                user = db.authenticate(
                    username,
                    password,
                )

                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.login_error = None

                    conversations = db.get_conversations(user["id"])

                    if conversations:
                        st.session_state.conversation_id = conversations[0]["id"]
                    else:
                        st.session_state.conversation_id = db.create_conversation(
                            user["id"],
                            "New Conversation",
                        )

                    st.session_state.messages = []

                    st.rerun()

                else:
                    st.error("Invalid username or password.")

            except Exception as exc:
                st.error("Authentication failed.")
                st.exception(exc)

        st.divider()

        st.caption("Don't have an account?")

        if st.button(
            "Create an account",
            use_container_width=True,
            key="go_to_signup",
        ):
            st.session_state.auth_view = "signup"
            st.rerun()


# ============================================================
# SIGN UP
# ============================================================


def signup_screen():
    st.title("📄 Enterprise Document Intelligence")

    st.caption("Create an account to start uploading and querying documents.")

    st.divider()

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.subheader("Create account")

        with st.form(
            "signup_form",
            clear_on_submit=False,
        ):
            new_username = st.text_input(
                "Choose a username",
                autocomplete="username",
                key="signup_username",
            )

            new_password = st.text_input(
                "Choose a password",
                type="password",
                autocomplete="new-password",
                key="signup_password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                autocomplete="new-password",
                key="signup_confirm_password",
            )

            submitted = st.form_submit_button(
                "Sign up",
                use_container_width=True,
            )

        if submitted:
            if not new_username.strip():
                st.error("Choose a username.")
                return

            if not new_password:
                st.error("Choose a password.")
                return

            if len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
                return

            if new_password != confirm_password:
                st.error("Passwords do not match.")
                return

            try:
                result = db.create_user(
                    new_username,
                    new_password,
                )

                if result:
                    st.success("Account created. You can now sign in.")
                    st.session_state.auth_view = "login"
                    st.rerun()
                else:
                    st.error("Could not create account. That username may already be taken.")

            except Exception as exc:
                st.error("Account creation failed.")
                st.exception(exc)

        st.divider()

        st.caption("Already have an account?")

        if st.button(
            "Back to sign in",
            use_container_width=True,
            key="go_to_login",
        ):
            st.session_state.auth_view = "login"
            st.rerun()


# ============================================================
# LOGOUT
# ============================================================


def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.selected_document_ids = []
    st.session_state.page = "Chat"
    st.session_state.auth_view = "login"

    st.rerun()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================


def upload_documents(
    uploaded_files,
    user_id,
    chunking_method,
    chunk_size,
    chunk_overlap,
):
    if not uploaded_files:
        return

    engine = get_chunking_engine()

    model = None

    if PGVECTOR_ENABLED:
        try:
            model = get_model()
        except Exception:
            model = None

    progress = st.progress(0)

    status = st.empty()

    total = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name

        try:
            status.info(f"Processing {filename}...")

            file_bytes = uploaded_file.getvalue()

            file_type = get_file_type(filename)

            chunks = run_chunking(
                engine,
                file_bytes,
                filename,
                chunking_method,
                chunk_size,
                chunk_overlap,
            )

            chunks = normalize_chunks(chunks)

            if not chunks:
                raise RuntimeError("No chunks were produced.")

            metadata = {
                "filename": filename,
                "file_type": file_type,
                "chunk_count": len(chunks),
            }

            document_id = db.save_document(
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                file_bytes=file_bytes,
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
            )

            embedding_function = None

            if PGVECTOR_ENABLED and model is not None:
                embedding_function = lambda text: create_embedding(
                    model,
                    text,
                )

            db.save_chunks(
                document_id=document_id,
                chunks=chunks,
                embedding_function=embedding_function,
            )

            status.success(f"Processed {filename}: " f"{len(chunks)} chunks.")

        except Exception as exc:
            status.error(f"Failed to process {filename}: " f"{exc}")

        progress.progress((index + 1) / total)

    st.success("Document processing completed.")


# ============================================================
# DOCUMENT PAGE
# ============================================================


def documents_page():
    st.title("📚 Documents")

    user = st.session_state.user

    st.subheader("Upload documents")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "csv",
            "xlsx",
            "xls",
            "json",
            "html",
            "htm",
        ],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        chunking_method = st.selectbox(
            "Chunking method",
            [
                "Recursive",
                "Fixed Size",
                "Sentence",
                "Paragraph",
            ],
        )

    with col2:
        chunk_size = st.number_input(
            "Chunk size",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )

    with col3:
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=5000,
            value=150,
            step=50,
        )

    if st.button(
        "🚀 Process Documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Select at least one document.")
        else:
            upload_documents(
                uploaded_files,
                user["id"],
                chunking_method,
                int(chunk_size),
                int(chunk_overlap),
            )

    st.divider()

    st.subheader("Your documents")

    documents = db.get_documents(
        user_id=user["id"],
        is_admin=False,
    )

    if not documents:
        st.info("No documents uploaded yet.")
        return

    for doc_index, document in enumerate(documents):
        widget_suffix = f"{document['id']}_{doc_index}"

        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"**{document['filename']}**")

                st.caption(
                    f"{document.get('file_type', 'Unknown')} "
                    f"• {document.get('file_size', 0):,} bytes"
                )

            with col2:
                st.metric(
                    "Chunk size",
                    document.get("chunk_size") or "-",
                )

            with col3:
                st.metric(
                    "Overlap",
                    document.get("chunk_overlap") or "-",
                )

            with col4:
                if st.button(
                    "Delete",
                    key=f"delete_document_{widget_suffix}",
                ):
                    try:
                        db.delete_document(
                            document["id"],
                            user["id"],
                            False,
                        )

                        if document["id"] in (st.session_state.selected_document_ids):
                            st.session_state.selected_document_ids.remove(document["id"])

                        st.rerun()

                    except Exception as exc:
                        st.error(f"Could not delete document: {exc}")


# ============================================================
# RETRIEVAL METHOD
# ============================================================


def retrieval_options():
    options = [
        "PostgreSQL Full-Text",
    ]

    if PGVECTOR_ENABLED:
        options.extend(
            [
                "pgvector Semantic",
                "Hybrid (pgvector + PostgreSQL Full-Text)",
            ]
        )

    return options


# ============================================================
# BUILD RAG PROMPT
# ============================================================


def build_rag_prompt(
    question,
    search_results,
    history,
):
    context_blocks = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):
        filename = result.get(
            "filename",
            "Unknown document",
        )

        page = result.get("page")

        content = result.get(
            "content",
            "",
        )

        location = filename

        if page is not None:
            location += f", page {page}"

        context_blocks.append(f"[Source {index}: {location}]\n" f"{content}")

    context = "\n\n".join(context_blocks)

    history_text = ""

    if history:
        history_parts = []

        for message in history[-10:]:
            history_parts.append(
                f"{message.get('role', 'user').upper()}: " f"{message.get('content', '')}"
            )

        history_text = "\n".join(history_parts)

    return f"""
You are an enterprise document intelligence assistant.

Answer the user's question using the supplied document
context whenever possible.

Rules:
1. Do not invent facts.
2. If the answer is not supported by the context, clearly say
   that the available documents do not contain enough information.
3. Prefer precise, useful answers over unnecessary explanation.
4. When appropriate, identify the source document.
5. If multiple sources disagree, explicitly mention the conflict.
6. Use the conversation history only to understand context.
7. Do not claim to have accessed documents that are not in the
   supplied context.

Conversation history:
{history_text}

Document context:
{context}

User question:
{question}

Answer:
""".strip()


# ============================================================
# CHAT RESPONSE
# ============================================================


def generate_answer(
    question,
    user_id,
    selected_document_ids,
    chunk_types,
    top_k,
    retrieval_method,
):
    model = get_model()

    query_embedding = None

    if (
        retrieval_method
        in {
            "pgvector Semantic",
            "Hybrid (pgvector + PostgreSQL Full-Text)",
        }
        and PGVECTOR_ENABLED
    ):
        query_embedding = create_embedding(
            model,
            question,
        )

        if query_embedding is None:
            retrieval_method = "PostgreSQL Full-Text"

    results = db.search_chunks(
        user_id=user_id,
        query=question,
        selected_document_ids=selected_document_ids,
        chunk_types=chunk_types,
        top_k=top_k,
        retrieval_method=retrieval_method,
        query_embedding=query_embedding,
    )

    if not results:
        return (
            "I couldn't find relevant information " "in the selected documents.",
            [],
        )

    prompt = build_rag_prompt(
        question,
        results,
        st.session_state.messages,
    )

    answer = call_model(
        model,
        prompt,
    )

    return answer, results


# ============================================================
# CHAT PAGE
# ============================================================


def chat_page():
    user = st.session_state.user

    st.title("💬 Document Intelligence")

    st.caption("Ask questions about your uploaded documents.")

    documents = db.get_documents(
        user_id=user["id"],
        is_admin=False,
    )

    document_map = {document["id"]: document["filename"] for document in documents}

    with st.sidebar:
        st.subheader("Retrieval")

        selected_ids = st.multiselect(
            "Documents",
            options=list(document_map.keys()),
            default=[
                document_id
                for document_id in st.session_state.selected_document_ids
                if document_id in document_map
            ],
            format_func=lambda document_id: document_map[document_id],
        )

        st.session_state.selected_document_ids = selected_ids

        retrieval_method = st.selectbox(
            "Retrieval method",
            retrieval_options(),
        )

        top_k = st.slider(
            "Results",
            min_value=1,
            max_value=20,
            value=5,
        )

        chunk_type_options = [
            "Text",
            "Table",
            "Heading",
            "List",
        ]

        chunk_types = st.multiselect(
            "Chunk types",
            chunk_type_options,
        )

    conversation_id = st.session_state.conversation_id

    if conversation_id is None:
        conversation_id = db.create_conversation(
            user["id"],
            "New Conversation",
        )

        st.session_state.conversation_id = conversation_id

    messages = db.get_messages(
        conversation_id,
        user["id"],
    )

    if not st.session_state.messages:
        st.session_state.messages = messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        question = question.strip()

        if not question:
            st.stop()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        db.save_message(
            conversation_id,
            "user",
            question,
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    answer, sources = generate_answer(
                        question=question,
                        user_id=user["id"],
                        selected_document_ids=(selected_ids or None),
                        chunk_types=(chunk_types or None),
                        top_k=top_k,
                        retrieval_method=retrieval_method,
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):
                                filename = source.get(
                                    "filename",
                                    "Unknown",
                                )

                                page = source.get("page")

                                score = source.get("similarity_score")

                                label = f"{index}. " f"{filename}"

                                if page is not None:
                                    label += f" — page {page}"

                                if score is not None:
                                    try:
                                        label += f" — score " f"{float(score):.4f}"
                                    except Exception:
                                        pass

                                st.markdown(f"**{label}**")

                                st.write(
                                    source.get(
                                        "content",
                                        "",
                                    )
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    db.save_message(
                        conversation_id,
                        "assistant",
                        answer,
                    )

                    if len(st.session_state.messages) == 2:
                        title = question[:80]

                        db.update_conversation_title(
                            conversation_id,
                            user["id"],
                            title,
                        )

                except Exception as exc:
                    error_message = "I couldn't process that request."

                    st.error(error_message)

                    st.exception(exc)


# ============================================================
# CONVERSATION PAGE
# ============================================================


def conversations_page():
    user = st.session_state.user

    st.title("🗂️ Conversations")

    if st.button(
        "➕ New Conversation",
        type="primary",
    ):
        conversation_id = db.create_conversation(
            user["id"],
            "New Conversation",
        )

        st.session_state.conversation_id = conversation_id

        st.session_state.messages = []

        st.rerun()

    conversations = db.get_conversations(user["id"])

    if not conversations:
        st.info("No conversations yet.")
        return

    for conv_index, conversation in enumerate(conversations):
        widget_suffix = f"{conversation['id']}_{conv_index}"

        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 1, 1])

            with col1:
                title = conversation.get("title") or "New Conversation"

                st.markdown(f"**{title}**")

                created_at = conversation.get("created_at")

                if created_at:
                    st.caption(str(created_at))

            with col2:
                if st.button(
                    "Open",
                    key=(f"open_conversation_{widget_suffix}"),
                ):
                    st.session_state.conversation_id = conversation["id"]

                    st.session_state.messages = db.get_messages(
                        conversation["id"],
                        user["id"],
                    )

                    st.session_state.page = "Chat"

                    st.rerun()

            with col3:
                if st.button(
                    "Delete",
                    key=(f"delete_conversation_{widget_suffix}"),
                ):
                    db.delete_conversation(
                        conversation["id"],
                        user["id"],
                    )

                    if st.session_state.conversation_id == conversation["id"]:
                        st.session_state.conversation_id = None

                        st.session_state.messages = []

                    st.rerun()


# ============================================================
# ADMIN PAGE
# ============================================================


def admin_page():
    user = st.session_state.user

    if (
        str(
            user.get(
                "role",
                "user",
            )
        ).lower()
        != "admin"
    ):
        st.error("Administrator access is required.")
        return

    st.title("🛡️ Administration")

    tabs = st.tabs(
        [
            "Overview",
            "Users",
            "Documents",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tabs[0]:
        stats = db.get_admin_stats()

        columns = st.columns(5)

        metric_names = [
            ("Users", "users"),
            ("Documents", "documents"),
            ("Chunks", "chunks"),
            ("Conversations", "conversations"),
            ("Messages", "messages"),
        ]

        for column, (
            label,
            key,
        ) in zip(
            columns,
            metric_names,
        ):
            with column:
                st.metric(
                    label,
                    stats.get(
                        key,
                        0,
                    ),
                )

    # ========================================================
    # USERS
    # ========================================================

    with tabs[1]:
        st.subheader("User management")

        users = db.get_users()

        if users:
            for row_index, managed_user in enumerate(users):
                widget_suffix = f"{managed_user['id']}_{row_index}"

                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                    with col1:
                        st.markdown(f"**{managed_user['username']}**")

                        st.caption(f"ID: {managed_user['id']}")

                    with col2:
                        current_role = str(
                            managed_user.get(
                                "role",
                                "user",
                            )
                        ).lower()

                        new_role = st.selectbox(
                            "Role",
                            [
                                "user",
                                "admin",
                            ],
                            index=(1 if current_role == "admin" else 0),
                            key=(f"role_{widget_suffix}"),
                        )

                    with col3:
                        if st.button(
                            "Save role",
                            key=(f"save_role_{widget_suffix}"),
                        ):
                            try:
                                success, message = db.set_user_role(
                                    managed_user["id"],
                                    new_role,
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.warning(message)

                            except Exception as exc:
                                st.error(str(exc))

                    with col4:
                        with st.popover("Reset password"):
                            new_password = st.text_input(
                                "New password",
                                type="password",
                                key=(f"password_{widget_suffix}"),
                            )

                            confirm_password = st.text_input(
                                "Confirm password",
                                type="password",
                                key=(f"confirm_password_{widget_suffix}"),
                            )

                            if st.button(
                                "Reset",
                                key=(f"reset_{widget_suffix}"),
                            ):
                                if not new_password:
                                    st.error("Password cannot be empty.")

                                elif new_password != confirm_password:
                                    st.error("Passwords do not match.")

                                else:
                                    try:
                                        db.reset_user_password(
                                            managed_user["id"],
                                            new_password,
                                        )

                                        st.success("Password reset.")

                                    except Exception as exc:
                                        st.error(str(exc))

                        if st.button(
                            "Delete",
                            key=(f"delete_user_{widget_suffix}"),
                        ):
                            if managed_user["id"] == user["id"]:
                                st.error("You cannot delete your own account.")
                            else:
                                try:
                                    success, message = db.delete_user(managed_user["id"])

                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.warning(message)

                                except Exception as exc:
                                    st.error(str(exc))

        st.divider()

        st.subheader("Create user")

        with st.form(
            "create_user_form",
            clear_on_submit=True,
        ):
            new_username = st.text_input("Username")

            new_password = st.text_input(
                "Password",
                type="password",
            )

            create_user_button = st.form_submit_button(
                "Create user",
                type="primary",
            )

        if create_user_button:
            if not new_username.strip():
                st.error("Username is required.")

            elif not new_password:
                st.error("Password is required.")

            else:
                try:
                    result = db.create_user(
                        new_username,
                        new_password,
                    )

                    if result:
                        st.success("User created successfully.")
                        st.rerun()

                    else:
                        st.error("Could not create user. " "The username may already exist.")

                except Exception as exc:
                    st.error(str(exc))

    # ========================================================
    # DOCUMENTS
    # ========================================================

    with tabs[2]:
        st.subheader("All documents")

        documents = db.get_all_documents_admin()

        if not documents:
            st.info("No documents found.")
        else:
            dataframe = pd.DataFrame(documents)

            visible_columns = [
                column
                for column in [
                    "id",
                    "user_id",
                    "username",
                    "filename",
                    "file_type",
                    "file_size",
                    "chunking_method",
                    "chunk_size",
                    "chunk_overlap",
                    "created_at",
                ]
                if column in dataframe.columns
            ]

            st.dataframe(
                dataframe[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Delete document")

            document_options = {
                (
                    f"{document['id']} — "
                    f"{document['filename']} — "
                    f"{document.get('username') or 'Unknown user'}"
                ): document["id"]
                for document in documents
            }

            selected_document = st.selectbox(
                "Select document",
                options=list(document_options.keys()),
            )

            if st.button(
                "Delete selected document",
                type="secondary",
            ):
                document_id = document_options[selected_document]

                try:
                    deleted = db.delete_document(
                        document_id,
                        is_admin=True,
                    )

                    if deleted:
                        st.success("Document deleted.")
                        st.rerun()

                    else:
                        st.warning("Document was not found.")

                except Exception as exc:
                    st.error(str(exc))


# ============================================================
# SIDEBAR
# ============================================================


def application_sidebar():
    user = st.session_state.user

    with st.sidebar:
        st.title("📄 Enterprise RAG")

        st.caption(f"Signed in as " f"**{user.get('username', 'User')}**")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Chat",
                "Documents",
                "Conversations",
            ]
            + (
                ["Administration"]
                if str(
                    user.get(
                        "role",
                        "user",
                    )
                ).lower()
                == "admin"
                else []
            ),
            key="navigation_radio",
        )

        st.session_state.page = page

        st.divider()

        if st.button(
            "🚪 Sign out",
            use_container_width=True,
        ):
            logout()


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.authenticated:
    if st.session_state.auth_view == "signup":
        signup_screen()
    else:
        login_screen()
    st.stop()


application_sidebar()


# ============================================================
# ROUTING
# ============================================================

if st.session_state.page == "Chat":
    chat_page()

elif st.session_state.page == "Documents":
    documents_page()

elif st.session_state.page == "Conversations":
    conversations_page()

elif st.session_state.page == "Administration":
    admin_page()

else:
    chat_page()