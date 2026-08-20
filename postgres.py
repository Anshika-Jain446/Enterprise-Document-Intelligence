import json
from typing import Optional, List, Dict, Any

import psycopg
from psycopg.rows import dict_row

from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)


class PostgreSQLStore:

    def __init__(
        self,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
    ):

        self.host = (
            host
            or POSTGRES_HOST
        )

        self.port = (
            port
            or POSTGRES_PORT
            or 5432
        )

        self.database = (
            database
            or POSTGRES_DB
        )

        self.user = (
            user
            or POSTGRES_USER
        )

        self.password = (
            password
            or POSTGRES_PASSWORD
        )

        self._create_tables()

    # ========================================================
    # CONNECTION
    # ========================================================

    def _connect(self):

        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
            row_factory=dict_row,
        )

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def _create_tables(self):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    CREATE EXTENSION IF NOT EXISTS vector;
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGSERIAL PRIMARY KEY,

                        filename TEXT NOT NULL,

                        title TEXT,

                        author TEXT,

                        file_type TEXT,

                        file_size BIGINT,

                        metadata JSONB DEFAULT '{}'::jsonb,

                        created_at TIMESTAMPTZ
                            DEFAULT NOW()
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        id BIGSERIAL PRIMARY KEY,

                        document_id BIGINT
                            REFERENCES documents(id)
                            ON DELETE CASCADE,

                        chunk_id INTEGER NOT NULL,

                        chunk_type TEXT NOT NULL,

                        content TEXT NOT NULL,

                        page INTEGER,

                        tokens INTEGER,

                        metadata JSONB
                            DEFAULT '{}'::jsonb,

                        embedding vector(1536),

                        created_at TIMESTAMPTZ
                            DEFAULT NOW(),

                        UNIQUE (
                            document_id,
                            chunk_id
                        )
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_tables (
                        id BIGSERIAL PRIMARY KEY,

                        document_id BIGINT
                            REFERENCES documents(id)
                            ON DELETE CASCADE,

                        chunk_id INTEGER,

                        page INTEGER,

                        sheet TEXT,

                        table_index INTEGER,

                        data JSONB,

                        content TEXT,

                        created_at TIMESTAMPTZ
                            DEFAULT NOW()
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    chunks_document_id_idx
                    ON chunks(document_id);
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    chunks_type_idx
                    ON chunks(chunk_type);
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    documents_filename_idx
                    ON documents(filename);
                    """
                )

            conn.commit()

    # ========================================================
    # DOCUMENT
    # ========================================================

    def add_document(
        self,
        filename,
        title=None,
        author=None,
        file_type=None,
        file_size=None,
        metadata=None,
    ):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    INSERT INTO documents (
                        filename,
                        title,
                        author,
                        file_type,
                        file_size,
                        metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING id;
                    """,
                    (
                        filename,
                        title,
                        author,
                        file_type,
                        file_size,
                        json.dumps(
                            metadata or {}
                        ),
                    ),
                )

                document_id = cur.fetchone()["id"]

            conn.commit()

        return document_id

    # ========================================================
    # DOCUMENT LOOKUP
    # ========================================================

    def get_document(
        self,
        document_id,
    ):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE id = %s;
                    """,
                    (document_id,),
                )

                return cur.fetchone()

    # ========================================================
    # INSERT CHUNKS
    # ========================================================

    def add_chunks(
        self,
        document_id,
        chunks,
    ):

        if not chunks:
            return 0

        inserted = 0

        with self._connect() as conn:

            with conn.cursor() as cur:

                for chunk in chunks:

                    if not isinstance(
                        chunk,
                        dict,
                    ):
                        continue

                    content = str(
                        chunk.get(
                            "content",
                            "",
                        )
                    ).strip()

                    if not content:
                        continue

                    metadata = chunk.get(
                        "metadata",
                        {},
                    )

                    embedding = chunk.get(
                        "embedding"
                    )

                    embedding_value = None

                    if embedding is not None:

                        embedding_value = (
                            self._vector_string(
                                embedding
                            )
                        )

                    cur.execute(
                        """
                        INSERT INTO chunks (
                            document_id,
                            chunk_id,
                            chunk_type,
                            content,
                            page,
                            tokens,
                            metadata,
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
                            %s
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
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                        """,
                        (
                            document_id,

                            chunk.get(
                                "chunk_id"
                            ),

                            chunk.get(
                                "chunk_type",
                                "Unknown",
                            ),

                            content,

                            chunk.get(
                                "page"
                            ),

                            chunk.get(
                                "tokens"
                            ),

                            json.dumps(
                                metadata
                            ),

                            embedding_value,
                        ),
                    )

                    inserted += 1

            conn.commit()

        return inserted

    # ========================================================
    # TABLES
    # ========================================================

    def add_table_chunks(
        self,
        document_id,
        chunks,
    ):

        if not chunks:
            return 0

        inserted = 0

        with self._connect() as conn:

            with conn.cursor() as cur:

                for chunk in chunks:

                    if (
                        not isinstance(
                            chunk,
                            dict,
                        )
                        or chunk.get(
                            "chunk_type"
                        ) != "Table"
                    ):
                        continue

                    cur.execute(
                        """
                        INSERT INTO document_tables (
                            document_id,
                            chunk_id,
                            page,
                            sheet,
                            table_index,
                            data,
                            content
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        );
                        """,
                        (
                            document_id,

                            chunk.get(
                                "chunk_id"
                            ),

                            chunk.get(
                                "page"
                            ),

                            chunk.get(
                                "metadata",
                                {},
                            ).get(
                                "sheet"
                            ),

                            chunk.get(
                                "metadata",
                                {},
                            ).get(
                                "table_index"
                            ),

                            json.dumps(
                                chunk.get(
                                    "table",
                                    [],
                                )
                            ),

                            chunk.get(
                                "content",
                                "",
                            ),
                        ),
                    )

                    inserted += 1

            conn.commit()

        return inserted

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        self,
        embedding,
        limit=10,
        document_id=None,
    ):

        if not embedding:
            return []

        vector = self._vector_string(
            embedding
        )

        with self._connect() as conn:

            with conn.cursor() as cur:

                if document_id is not None:

                    cur.execute(
                        """
                        SELECT
                            id,
                            document_id,
                            chunk_id,
                            chunk_type,
                            content,
                            page,
                            metadata,
                            1 - (
                                embedding <=> %s::vector
                            ) AS similarity
                        FROM chunks
                        WHERE embedding IS NOT NULL
                          AND document_id = %s
                        ORDER BY
                            embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (
                            vector,
                            document_id,
                            vector,
                            limit,
                        ),
                    )

                else:

                    cur.execute(
                        """
                        SELECT
                            id,
                            document_id,
                            chunk_id,
                            chunk_type,
                            content,
                            page,
                            metadata,
                            1 - (
                                embedding <=> %s::vector
                            ) AS similarity
                        FROM chunks
                        WHERE embedding IS NOT NULL
                        ORDER BY
                            embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (
                            vector,
                            vector,
                            limit,
                        ),
                    )

                return cur.fetchall()

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

    def document_search(
        self,
        query,
        limit=10,
    ):

        if not query or not query.strip():
            return []

        search_query = query.strip()

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.chunk_id,
                        c.chunk_type,
                        c.content,
                        c.page,
                        c.metadata,
                        d.filename,
                        d.title,
                        ts_rank(
                            to_tsvector(
                                'english',
                                c.content
                            ),
                            plainto_tsquery(
                                'english',
                                %s
                            )
                        ) AS rank
                    FROM chunks c
                    JOIN documents d
                        ON d.id = c.document_id
                    WHERE
                        to_tsvector(
                            'english',
                            c.content
                        )
                        @@ plainto_tsquery(
                            'english',
                            %s
                        )
                        OR d.filename ILIKE %s
                        OR d.title ILIKE %s
                    ORDER BY rank DESC
                    LIMIT %s;
                    """,
                    (
                        search_query,
                        search_query,
                        f"%{search_query}%",
                        f"%{search_query}%",
                        limit,
                    ),
                )

                return cur.fetchall()

    # ========================================================
    # TABLE SEARCH
    # ========================================================

    def table_search(
        self,
        query,
        limit=10,
    ):

        if not query or not query.strip():
            return []

        search_query = query.strip()

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        id,
                        document_id,
                        chunk_id,
                        page,
                        sheet,
                        table_index,
                        data,
                        content
                    FROM document_tables
                    WHERE
                        content ILIKE %s
                    ORDER BY id DESC
                    LIMIT %s;
                    """,
                    (
                        f"%{search_query}%",
                        limit,
                    ),
                )

                return cur.fetchall()

    # ========================================================
    # GET DOCUMENT CHUNKS
    # ========================================================

    def get_chunks(
        self,
        document_id,
        limit=1000,
    ):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        *
                    FROM chunks
                    WHERE document_id = %s
                    ORDER BY chunk_id
                    LIMIT %s;
                    """,
                    (
                        document_id,
                        limit,
                    ),
                )

                return cur.fetchall()

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id,
    ):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s;
                    """,
                    (document_id,),
                )

                deleted = cur.rowcount

            conn.commit()

        return deleted > 0

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def test_connection(self):

        with self._connect() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    "SELECT 1 AS status;"
                )

                result = cur.fetchone()

        return result["status"] == 1

    # ========================================================
    # VECTOR HELPER
    # ========================================================

    @staticmethod
    def _vector_string(
        embedding,
    ):

        if hasattr(
            embedding,
            "tolist",
        ):
            embedding = embedding.tolist()

        if not isinstance(
            embedding,
            (list, tuple),
        ):
            raise ValueError(
                "Embedding must be a list or tuple."
            )

        return "[" + ",".join(
            str(float(value))
            for value in embedding
        ) + "]"