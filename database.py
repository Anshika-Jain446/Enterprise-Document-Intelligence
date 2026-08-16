import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash


class Database:

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        self.database_url = st.secrets.get(
            "DATABASE_URL"
        )

        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured."
            )

    # ============================================================
    # CONNECTION
    # ============================================================

    def get_connection(self):

        return psycopg2.connect(
            self.database_url
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

                result = cursor.fetchone()

                return result is not None

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
    # GET USER DOCUMENTS
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

                connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()