import os
from urllib.parse import urlparse

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)


class Database:
    """
    Railway PostgreSQL database layer.

    Uses:
        1. Streamlit Secrets
        2. Environment variables

    PostgreSQL only.
    No Supabase.
    """

    APPLICATION_NAME = "EnterpriseDocumentIntelligence"

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        self.database_url = self._secret(
            "DATABASE_URL",
            "",
        )

        self.postgres_host = self._secret(
            "POSTGRES_HOST",
            "",
        )

        self.postgres_port = self._secret(
            "POSTGRES_PORT",
            "5432",
        )

        self.postgres_db = self._secret(
            "POSTGRES_DB",
            "postgres",
        )

        self.postgres_user = self._secret(
            "POSTGRES_USER",
            "postgres",
        )

        self.postgres_password = self._secret(
            "POSTGRES_PASSWORD",
            "",
        )

    # ============================================================
    # STREAMLIT SECRETS + ENVIRONMENT
    # ============================================================

    @staticmethod
    def _secret(
        name,
        default="",
    ):
        """
        Read configuration in this order:

        1. Streamlit Secrets
        2. Environment variable
        3. Default value
        """

        # --------------------------------------------------------
        # Streamlit Secrets
        # --------------------------------------------------------

        try:

            value = st.secrets.get(
                name,
                None,
            )

            if value is not None:

                value = str(value).strip()

                if value:
                    return value

        except Exception:
            pass

        # --------------------------------------------------------
        # Environment variable
        # --------------------------------------------------------

        value = os.getenv(name)

        if value is not None:

            value = str(value).strip()

            if value:
                return value

        return default

    # ============================================================
    # CLEAN DATABASE URL
    # ============================================================

    @staticmethod
    def _clean_database_url(
        url,
    ):

        if not url:
            return ""

        url = str(url).strip()

        # Remove accidental double quotes
        if (
            len(url) >= 2
            and url[0] == '"'
            and url[-1] == '"'
        ):
            url = url[1:-1]

        # Remove accidental single quotes
        if (
            len(url) >= 2
            and url[0] == "'"
            and url[-1] == "'"
        ):
            url = url[1:-1]

        return url.strip()

    # ============================================================
    # DATABASE CONNECTION
    # ============================================================

    def get_connection(self):

        database_url = self._clean_database_url(
            self.database_url
        )

        # --------------------------------------------------------
        # METHOD 1: DATABASE_URL
        # --------------------------------------------------------

        if database_url:

            try:

                return psycopg2.connect(
                    database_url,
                    connect_timeout=20,
                    sslmode="require",
                    application_name=self.APPLICATION_NAME,
                )

            except psycopg2.OperationalError as exc:

                raise RuntimeError(
                    self._connection_error(
                        exc,
                        database_url,
                    )
                ) from exc

        # --------------------------------------------------------
        # METHOD 2: INDIVIDUAL POSTGRESQL SETTINGS
        # --------------------------------------------------------

        if not self.postgres_host:

            raise RuntimeError(
                "POSTGRES_HOST is not configured."
            )

        if not self.postgres_password:

            raise RuntimeError(
                "POSTGRES_PASSWORD is not configured."
            )

        try:

            return psycopg2.connect(
                host=self.postgres_host,
                port=int(self.postgres_port),
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password,
                connect_timeout=20,
                sslmode="require",
                application_name=self.APPLICATION_NAME,
            )

        except psycopg2.OperationalError as exc:

            raise RuntimeError(
                self._connection_error(
                    exc,
                    None,
                )
            ) from exc

    # ============================================================
    # CONNECTION ERROR
    # ============================================================

    def _connection_error(
        self,
        exc,
        database_url=None,
    ):

        error = str(exc)

        if (
            "password authentication failed"
            in error.lower()
        ):

            return (
                "PostgreSQL password authentication failed.\n\n"
                "Check DATABASE_URL or "
                "POSTGRES_PASSWORD.\n\n"
                "Use the PostgreSQL database password, "
                "not a Railway dashboard password."
            )

        if database_url:

            try:

                parsed = urlparse(
                    database_url
                )

                return (
                    "Railway PostgreSQL connection failed.\n\n"
                    f"Host: {parsed.hostname}\n"
                    f"Port: {parsed.port or 5432}\n"
                    f"Database: "
                    f"{parsed.path.lstrip('/')}\n"
                    f"User: {parsed.username}\n\n"
                    f"Error: {error}"
                )

            except Exception:

                return (
                    "Railway PostgreSQL connection failed.\n\n"
                    f"Error: {error}"
                )

        return (
            "Railway PostgreSQL connection failed.\n\n"
            f"Host: {self.postgres_host}\n"
            f"Port: {self.postgres_port}\n"
            f"Database: {self.postgres_db}\n"
            f"User: {self.postgres_user}\n\n"
            f"Error: {error}"
        )

    # ============================================================
    # TEST CONNECTION
    # ============================================================

    def test_connection(self):

        connection = self.get_connection()

        try:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SELECT 1;"
                )

                return (
                    cursor.fetchone()
                    is not None
                )

        finally:

            connection.close()

    # ============================================================
    # CREATE USER
    # ============================================================

    def create_user(
        self,
        username,
        full_name,
        email,
        password,
        role="User",
    ):

        if not username:
            raise ValueError(
                "Username is required."
            )

        if not email:
            raise ValueError(
                "Email is required."
            )

        if not password:
            raise ValueError(
                "Password is required."
            )

        password_hash = generate_password_hash(
            password
        )

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    INSERT INTO users
                    (
                        username,
                        full_name,
                        email,
                        password_hash,
                        role
                    )
                    VALUES
                    (%s, %s, %s, %s, %s)
                    RETURNING
                        id,
                        username,
                        full_name,
                        email,
                        role;
                    """,
                    (
                        username,
                        full_name,
                        email,
                        password_hash,
                        role,
                    ),
                )

                user = cursor.fetchone()

                connection.commit()

                return user

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ============================================================
    # CHECK USER EXISTS
    # ============================================================

    def user_exists(
        self,
        username,
        email,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT 1
                    FROM users
                    WHERE username = %s
                       OR email = %s
                    LIMIT 1;
                    """,
                    (
                        username,
                        email,
                    ),
                )

                return (
                    cursor.fetchone()
                    is not None
                )

        finally:

            connection.close()

    # ============================================================
    # GET USER
    # ============================================================

    def get_user(
        self,
        username,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    SELECT
                        id,
                        username,
                        full_name,
                        email,
                        password_hash,
                        role
                    FROM users
                    WHERE username = %s
                    LIMIT 1;
                    """,
                    (
                        username,
                    ),
                )

                return cursor.fetchone()

        finally:

            connection.close()

    # ============================================================
    # GET USER BY ID
    # ============================================================

    def get_user_by_id(
        self,
        user_id,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    SELECT
                        id,
                        username,
                        full_name,
                        email,
                        password_hash,
                        role
                    FROM users
                    WHERE id = %s
                    LIMIT 1;
                    """,
                    (
                        user_id,
                    ),
                )

                return cursor.fetchone()

        finally:

            connection.close()

    # ============================================================
    # AUTHENTICATE USER
    # ============================================================

    def authenticate_user(
        self,
        username,
        password,
    ):

        user = self.get_user(
            username
        )

        if not user:
            return None

        password_hash = user.get(
            "password_hash"
        )

        if not password_hash:
            return None

        if not check_password_hash(
            password_hash,
            password,
        ):
            return None

        return user

    # ============================================================
    # AUTHENTICATE ALIAS
    # ============================================================

    def authenticate(
        self,
        username,
        password,
    ):

        return self.authenticate_user(
            username,
            password,
        )

    # ============================================================
    # GET ALL USERS
    # ============================================================

    def get_all_users(self):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    SELECT
                        id,
                        username,
                        full_name,
                        email,
                        role
                    FROM users
                    ORDER BY id DESC;
                    """
                )

                return cursor.fetchall()

        finally:

            connection.close()

    # ============================================================
    # DELETE USER
    # ============================================================

    def delete_user(
        self,
        username,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    DELETE FROM users
                    WHERE username = %s;
                    """,
                    (
                        username,
                    ),
                )

                connection.commit()

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ============================================================
    # CREATE DOCUMENT
    # ============================================================

    def create_document(
        self,
        filename,
        uploaded_by,
        file_path=None,
        file_type=None,
        file_size=None,
        status="uploaded",
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    INSERT INTO documents
                    (
                        filename,
                        uploaded_by,
                        file_path,
                        file_type,
                        file_size,
                        status
                    )
                    VALUES
                    (%s, %s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (
                        filename,
                        uploaded_by,
                        file_path,
                        file_type,
                        file_size,
                        status,
                    ),
                )

                document = cursor.fetchone()

                connection.commit()

                return document

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ============================================================
    # GET ALL DOCUMENTS
    # ============================================================

    def get_all_documents(self):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    SELECT
                        d.id,
                        d.filename,
                        d.uploaded_by,
                        d.file_path,
                        d.file_type,
                        d.file_size,
                        d.uploaded_at,
                        d.status,
                        u.username,
                        u.full_name,
                        u.email
                    FROM documents d
                    LEFT JOIN users u
                        ON d.uploaded_by = u.id
                    ORDER BY
                        d.uploaded_at DESC;
                    """
                )

                return cursor.fetchall()

        finally:

            connection.close()

    # ============================================================
    # GET DOCUMENTS
    # ============================================================

    def get_documents(
        self,
        user_id=None,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                if user_id is None:

                    cursor.execute(
                        """
                        SELECT
                            d.id,
                            d.filename,
                            d.uploaded_by,
                            d.file_path,
                            d.file_type,
                            d.file_size,
                            d.uploaded_at,
                            d.status,
                            u.username,
                            u.full_name,
                            u.email
                        FROM documents d
                        LEFT JOIN users u
                            ON d.uploaded_by = u.id
                        ORDER BY
                            d.uploaded_at DESC;
                        """
                    )

                else:

                    cursor.execute(
                        """
                        SELECT
                            id,
                            filename,
                            uploaded_by,
                            file_path,
                            file_type,
                            file_size,
                            uploaded_at,
                            status
                        FROM documents
                        WHERE uploaded_by = %s
                        ORDER BY uploaded_at DESC;
                        """,
                        (
                            user_id,
                        ),
                    )

                return cursor.fetchall()

        finally:

            connection.close()

    # ============================================================
    # GET USER DOCUMENTS
    # ============================================================

    def get_user_documents(
        self,
        user_id,
    ):

        return self.get_documents(
            user_id=user_id
        )

    # ============================================================
    # GET DOCUMENT
    # ============================================================

    def get_document(
        self,
        document_id,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                cursor.execute(
                    """
                    SELECT
                        id,
                        filename,
                        uploaded_by,
                        file_path,
                        file_type,
                        file_size,
                        uploaded_at,
                        status
                    FROM documents
                    WHERE id = %s
                    LIMIT 1;
                    """,
                    (
                        document_id,
                    ),
                )

                return cursor.fetchone()

        finally:

            connection.close()

    # ============================================================
    # GET DOCUMENT BY ID
    # ============================================================

    def get_document_by_id(
        self,
        document_id,
        user_id=None,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:

                if user_id is None:

                    cursor.execute(
                        """
                        SELECT *
                        FROM documents
                        WHERE id = %s
                        LIMIT 1;
                        """,
                        (
                            document_id,
                        ),
                    )

                else:

                    cursor.execute(
                        """
                        SELECT *
                        FROM documents
                        WHERE id = %s
                          AND uploaded_by = %s
                        LIMIT 1;
                        """,
                        (
                            document_id,
                            user_id,
                        ),
                    )

                return cursor.fetchone()

        finally:

            connection.close()

    # ============================================================
    # GET USER DOCUMENT
    # ============================================================

    def get_user_document(
        self,
        document_id,
        user_id,
    ):

        return self.get_document_by_id(
            document_id,
            user_id,
        )

    # ============================================================
    # DELETE DOCUMENT
    # ============================================================

    def delete_document(
        self,
        document_id,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s;
                    """,
                    (
                        document_id,
                    ),
                )

                deleted = (
                    cursor.rowcount > 0
                )

                connection.commit()

                return deleted

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ============================================================
    # DELETE USER DOCUMENT
    # ============================================================

    def delete_user_document(
        self,
        document_id,
        user_id,
    ):

        connection = self.get_connection()

        try:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE id = %s
                      AND uploaded_by = %s;
                    """,
                    (
                        document_id,
                        user_id,
                    ),
                )

                deleted = (
                    cursor.rowcount > 0
                )

                connection.commit()

                return deleted

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()