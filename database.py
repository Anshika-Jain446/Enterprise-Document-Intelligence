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
        # PostgreSQL
        # --------------------------------------------------------

        self.database_url = st.secrets.get(
            "DATABASE_URL"
        )

        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured."
            )

        # --------------------------------------------------------
        # Supabase
        #
        # IMPORTANT:
        # Supabase is NOT required for login/authentication.
        # PostgreSQL is the authentication database.
        # --------------------------------------------------------

        self.supabase_url = st.secrets.get(
            "SUPABASE_URL",
            ""
        )

        self.supabase_service_key = st.secrets.get(
            "SUPABASE_SERVICE_ROLE_KEY",
            ""
        )

        # --------------------------------------------------------
        # Storage bucket
        # --------------------------------------------------------

        self.storage_bucket = (
            "enterprise-documents"
        )

        # --------------------------------------------------------
        # Supabase client
        # --------------------------------------------------------

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
    # CONNECTION
    # ============================================================

    def get_connection(self):

        return psycopg2.connect(
            self.database_url
        )

    # ============================================================
    # TEST DATABASE CONNECTION
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
    # SUPABASE AVAILABILITY
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
                        role,
                        created_at;
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
                        role,
                        created_at
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
    # AUTHENTICATE USER
    #
    # This ONLY depends on PostgreSQL.
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
                        role,
                        created_at
                    FROM users
                    ORDER BY created_at DESC;
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
        """
        Upload a physical document to Supabase Storage.

        The file remains in Storage until explicitly deleted.
        """

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
        """
        Permanently remove a physical document
        from Supabase Storage.
        """

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
    # DOWNLOAD DOCUMENT FROM STORAGE
    # ============================================================

    def download_document_file(
        self,
        storage_path,
    ):
        """
        Download a private document directly from
        Supabase Storage.

        Returns bytes.
        """

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
        """
        Create a temporary signed URL for a private document.
        """

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
                result.get(
                    "signedURL"
                )
                or
                result.get(
                    "signedUrl"
                )
                or
                result.get(
                    "signed_url"
                )
            )

        return None

    # ============================================================
    # CREATE DOCUMENT RECORD
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
    #
    # 1. Upload file to Supabase.
    # 2. Save metadata to PostgreSQL.
    # 3. Roll back Storage upload if DB fails.
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

        # --------------------------------------------------------
        # Unique storage path
        #
        # Prevents collisions when the same user uploads
        # the same filename more than once.
        # --------------------------------------------------------

        safe_filename = os.path.basename(
            filename
        )

        unique_id = uuid.uuid4().hex

        storage_path = (
            f"{uploaded_by}/"
            f"{unique_id}_"
            f"{safe_filename}"
        )

        # --------------------------------------------------------
        # Upload physical file
        # --------------------------------------------------------

        self.upload_document_file(
            file_bytes=file_bytes,
            storage_path=storage_path,
            content_type=(
                file_type
                or "application/octet-stream"
            ),
        )

        try:

            # ----------------------------------------------------
            # Save metadata
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Database failed.
            # Remove orphaned Storage file.
            # ----------------------------------------------------

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
    # GET USER DOCUMENTS
    #
    # IMPORTANT:
    # This is how users see their OLD documents.
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
    #
    # Security helper:
    # Only returns a document belonging to the given user.
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
    #
    # Deletes BOTH:
    # 1. Supabase Storage file
    # 2. PostgreSQL metadata
    # ============================================================

    def delete_document(
        self,
        document_id,
    ):

        connection = self.get_connection()

        try:

            # ----------------------------------------------------
            # Get document information
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Delete physical Storage file
            # ----------------------------------------------------

            storage_path = document.get(
                "file_path"
            )

            if storage_path:

                self.delete_document_file(
                    storage_path
                )

            # ----------------------------------------------------
            # Delete PostgreSQL record
            # ----------------------------------------------------

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
    #
    # Security helper:
    # A normal user can only delete their own document.
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