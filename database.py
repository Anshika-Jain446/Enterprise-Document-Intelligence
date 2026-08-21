import os
import uuid

import streamlit as st
import psycopg2

from psycopg2.extras import RealDictCursor

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from supabase import create_client


class Database:

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        # --------------------------------------------------------
        # DATABASE_URL
        # --------------------------------------------------------

        self.database_url = self._secret(
            "DATABASE_URL",
            "",
        )

        # --------------------------------------------------------
        # INDIVIDUAL POSTGRESQL SETTINGS
        # --------------------------------------------------------

        self.postgres_host = self._secret(
            "POSTGRES_HOST",
            "aws-0-ap-south-1.pooler.supabase.com",
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

        # --------------------------------------------------------
        # SUPABASE STORAGE
        # --------------------------------------------------------

        self.supabase_url = self._secret(
            "SUPABASE_URL",
            "",
        )

        self.supabase_service_key = self._secret(
            "SUPABASE_SERVICE_ROLE_KEY",
            "",
        )

        self.storage_bucket = "enterprise-documents"

        self.supabase = None

        if (
            self.supabase_url
            and self.supabase_service_key
        ):
            self.supabase = create_client(
                self.supabase_url,
                self.supabase_service_key,
            )

    # ============================================================
    # READ SECRET
    # ============================================================

    def _secret(
        self,
        name,
        default="",
    ):

        # First try Streamlit Secrets
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

        # Then try environment variables
        value = os.getenv(name)

        if value:

            return str(value).strip()

        return default

    # ============================================================
    # CLEAN DATABASE URL
    # ============================================================

    def _clean_database_url(
        self,
        url,
    ):

        if not url:
            return ""

        url = str(url).strip()

        # Remove accidental surrounding quotes
        if (
            len(url) >= 2
            and url[0] == '"'
            and url[-1] == '"'
        ):
            url = url[1:-1]

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
                    application_name=(
                        "EnterpriseDocumentIntelligence"
                    ),
                )

            except psycopg2.OperationalError:
                # If DATABASE_URL fails, continue to
                # individual PostgreSQL settings.
                pass

        # --------------------------------------------------------
        # METHOD 2: INDIVIDUAL POSTGRESQL SETTINGS
        # --------------------------------------------------------

        if not self.postgres_password:

            raise RuntimeError(
                "PostgreSQL password is missing.\n\n"
                "Configure either DATABASE_URL or:\n\n"
                "POSTGRES_HOST\n"
                "POSTGRES_PORT\n"
                "POSTGRES_DB\n"
                "POSTGRES_USER\n"
                "POSTGRES_PASSWORD"
            )

        try:

            connection = psycopg2.connect(
                host=self.postgres_host,
                port=int(self.postgres_port),
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password,
                connect_timeout=20,
                sslmode="require",
                application_name=(
                    "EnterpriseDocumentIntelligence"
                ),
            )

            return connection

        except psycopg2.OperationalError as exc:

            error = str(exc)

            if (
                "password authentication failed"
                in error.lower()
            ):

                raise RuntimeError(
                    "Supabase rejected the PostgreSQL "
                    "password.\n\n"
                    "Check POSTGRES_PASSWORD or "
                    "DATABASE_URL in Streamlit Secrets.\n\n"
                    "IMPORTANT:\n"
                    "This must be the PostgreSQL database "
                    "password from Supabase, NOT your "
                    "Supabase dashboard password."
                ) from exc

            raise RuntimeError(
                "PostgreSQL connection failed.\n\n"
                f"Host: {self.postgres_host}\n"
                f"Port: {self.postgres_port}\n"
                f"Database: {self.postgres_db}\n"
                f"User: {self.postgres_user}\n\n"
                f"Error: {error}"
            ) from exc

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

                result = cursor.fetchone()

                return result is not None

        finally:

            connection.close()

    # ============================================================
    # STORAGE AVAILABLE
    # ============================================================

    def storage_available(self):

        return self.supabase is not None

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
    # GET USER BY USERNAME
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
    # AUTHENTICATE
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
    # SUPABASE STORAGE
    # ============================================================

    def upload_document_file(
        self,
        file_bytes,
        storage_path,
        content_type="application/octet-stream",
    ):

        if not self.supabase:

            raise RuntimeError(
                "Supabase Storage is not configured."
            )

        if not file_bytes:

            raise ValueError(
                "File is empty."
            )

        result = (
            self.supabase
            .storage
            .from_(
                self.storage_bucket
            )
            .upload(
                storage_path,
                file_bytes,
                {
                    "content-type": content_type,
                    "upsert": "false",
                },
            )
        )

        return result

    # ============================================================
    # DELETE FILE FROM SUPABASE STORAGE
    # ============================================================

    def delete_document_file(
        self,
        storage_path,
    ):

        if not storage_path:
            return False

        if not self.supabase:
            return False

        result = (
            self.supabase
            .storage
            .from_(
                self.storage_bucket
            )
            .remove(
                [storage_path]
            )
        )

        return result

    # ============================================================
    # DOWNLOAD FILE FROM SUPABASE STORAGE
    # ============================================================

    def download_document_file(
        self,
        storage_path,
    ):

        if not storage_path:
            return None

        if not self.supabase:

            raise RuntimeError(
                "Supabase Storage is not configured."
            )

        result = (
            self.supabase
            .storage
            .from_(
                self.storage_bucket
            )
            .download(
                storage_path
            )
        )

        return result

    # ============================================================
    # CREATE TEMPORARY DOWNLOAD URL
    # ============================================================

    def create_document_download_url(
        self,
        storage_path,
        expires_in=3600,
    ):

        if not storage_path:
            return None

        if not self.supabase:
            return None

        result = (
            self.supabase
            .storage
            .from_(
                self.storage_bucket
            )
            .create_signed_url(
                storage_path,
                expires_in,
            )
        )

        if isinstance(
            result,
            dict,
        ):

            return (
                result.get("signedURL")
                or result.get("signedUrl")
                or result.get("signed_url")
            )

        return None

    # ============================================================
    # CREATE DOCUMENT DATABASE RECORD
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
    # SAVE DOCUMENT
    # ============================================================

    def save_document(
        self,
        file_bytes,
        filename,
        uploaded_by,
        file_type=None,
        file_size=None,
    ):

        if not file_bytes:

            raise ValueError(
                "File is empty."
            )

        if not filename:

            raise ValueError(
                "Filename is required."
            )

        if not uploaded_by:

            raise ValueError(
                "uploaded_by is required."
            )

        if not self.supabase:

            raise RuntimeError(
                "Supabase Storage is not configured."
            )

        safe_filename = os.path.basename(
            filename
        )

        unique_id = uuid.uuid4().hex

        storage_path = (
            f"{uploaded_by}/"
            f"{unique_id}_"
            f"{safe_filename}"
        )

        self.upload_document_file(
            file_bytes=file_bytes,
            storage_path=storage_path,
            content_type=(
                file_type
                or "application/octet-stream"
            ),
        )

        try:

            document = self.create_document(
                filename=safe_filename,
                uploaded_by=uploaded_by,
                file_path=storage_path,
                file_type=file_type,
                file_size=file_size,
                status="uploaded",
            )

            return document

        except Exception:

            try:

                self.delete_document_file(
                    storage_path
                )

            except Exception:

                pass

            raise

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
    # GET USER'S DOCUMENTS
    # ============================================================

    def get_user_documents(
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
    # GET SINGLE DOCUMENT
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
    # GET USER DOCUMENT
    # ============================================================

    def get_user_document(
        self,
        document_id,
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
                        filename,
                        uploaded_by,
                        file_path,
                        file_type,
                        file_size,
                        uploaded_at,
                        status
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
    # DELETE DOCUMENT
    # ============================================================

    def delete_document(
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
                        file_path
                    FROM documents
                    WHERE id = %s
                    LIMIT 1;
                    """,
                    (
                        document_id,
                    ),
                )

                document = cursor.fetchone()

            if not document:
                return False

            storage_path = document.get(
                "file_path"
            )

            if storage_path:

                self.delete_document_file(
                    storage_path
                )

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

            connection.commit()

            return True

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

        document = self.get_user_document(
            document_id,
            user_id,
        )

        if not document:
            return False

        return self.delete_document(
            document_id
        )