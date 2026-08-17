import os
import tempfile
import hashlib
import mimetypes

import pandas as pd
import streamlit as st

from extractor import DocumentExtractor
from chunking import ChunkingEngine
from vector_db import VectorDatabase
from llm import GeminiLLM
from auth import Authentication
from register import Register
from database import Database
from agent import EnterpriseRAGAgent

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    SUPPORTED_FILES,
    LLM_MODEL,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Document Intelligence System",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "logged_in": False,
    "username": "",
    "role": "",
    "user_id": None,
    "full_name": "",
    "email": "",

    "chat_history": [],

    "db_created": False,
    "db_loaded": False,

    "all_documents": [],
    "combined_text": "",
    "combined_tables": [],

    "chunks": [],

    "processed_file_names": [],
    "indexed_file_names": [],

    "document_chunks": {},
}


for key, default_value in SESSION_DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# AUTHENTICATION OBJECTS
# ============================================================

try:

    auth = Authentication()
    register = Register()
    database = Database()

except Exception as e:

    st.error(
        f"Application initialization failed: {e}"
    )

    st.stop()


# ============================================================
# PERMANENT STORAGE HELPERS
# ============================================================

def make_storage_path(
    user_id,
    filename,
    file_bytes,
):
    """
    Creates a unique permanent Supabase Storage path.
    """

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    safe_filename = os.path.basename(
        filename
    )

    return (
        f"users/{user_id}/"
        f"{file_hash}_{safe_filename}"
    )


def get_content_type(
    filename,
):
    """
    Detect MIME type.
    """

    content_type, _ = mimetypes.guess_type(
        filename
    )

    return (
        content_type
        or "application/octet-stream"
    )


# ============================================================
# LOGIN / SIGN UP
# ============================================================

if not st.session_state.logged_in:

    st.title(
        "📄 Enterprise Document Intelligence System"
    )

    st.markdown(
        "### 🤖 Enterprise Agentic RAG Pipeline"
    )

    st.info(
        "Please login or create an account to continue."
    )

    page = st.radio(
        "Authentication",
        ["Login", "Sign Up"],
        horizontal=True,
    )

    if page == "Login":

        auth.login_page()

    else:

        register.register_page()

    st.stop()


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title(
    "📄 Enterprise Document Intelligence System"
)

st.markdown(
    "### 🤖 Enterprise Agentic RAG Pipeline"
)


# ============================================================
# VECTOR DATABASE
# ============================================================

if "vector_db" not in st.session_state:

    try:

        st.session_state.vector_db = (
            VectorDatabase()
        )

    except Exception as e:

        st.error(
            f"Failed to initialize Vector Database: {e}"
        )

        st.stop()


db = st.session_state.vector_db


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.success(
        f"👤 {st.session_state.username}"
    )

    st.caption(
        f"Role: {st.session_state.role}"
    )

    if st.session_state.full_name:

        st.caption(
            f"Name: {st.session_state.full_name}"
        )

    if st.session_state.email:

        st.caption(
            f"Email: {st.session_state.email}"
        )

    st.divider()

    st.header("⚙ Settings")

    chunk_method = st.selectbox(
        "Chunking Method",
        [
            "Character",
            "Recursive",
            "Token",
            "Markdown",
            "Context",
            "Table",
        ],
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=CHUNK_SIZE,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=CHUNK_OVERLAP,
        step=10,
    )

    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=5,
    )


    # ========================================================
    # ADMIN CONTROLS
    # ========================================================

    if st.session_state.role == "Admin":

        st.divider()

        st.subheader("🛡 Admin Controls")

        st.success(
            "You have administrator access."
        )


        # ====================================================
        # USER MANAGEMENT
        # ====================================================

        with st.expander(
            "👥 User Management",
            expanded=False,
        ):

            try:

                users = database.get_all_users()

                if not users:

                    st.info(
                        "No users found."
                    )

                else:

                    st.metric(
                        "Total Registered Users",
                        len(users),
                    )

                    st.divider()

                    for user in users:

                        st.markdown(
                            f"### 👤 "
                            f"{user.get('username', 'Unknown')}"
                        )

                        col_a, col_b = st.columns(2)

                        with col_a:

                            st.write(
                                f"**Full Name:** "
                                f"{user.get('full_name', 'Unknown')}"
                            )

                            st.write(
                                f"**Email:** "
                                f"{user.get('email', 'Unknown')}"
                            )

                        with col_b:

                            st.write(
                                f"**Role:** "
                                f"{user.get('role', 'User')}"
                            )

                            st.write(
                                f"**Created:** "
                                f"{user.get('created_at', 'Unknown')}"
                            )

                        st.divider()

            except Exception as e:

                st.error(
                    f"Unable to load users: {e}"
                )


        # ====================================================
        # VECTOR DATABASE INFORMATION
        # ====================================================

        with st.expander(
            "🗄 Vector Database Information",
            expanded=False,
        ):

            try:

                stats = db.get_stats()

                status = stats.get(
                    "status",
                    "Unknown"
                )

                if status == "Ready":

                    st.success(
                        "🟢 Vector Database Ready"
                    )

                elif status == "Not Created":

                    st.warning(
                        "🟡 Vector Database Not Created"
                    )

                else:

                    st.error(
                        "🔴 Vector Database Error"
                    )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Indexed Chunks",
                        stats.get(
                            "chunks",
                            0
                        ),
                    )

                with c2:

                    st.metric(
                        "Vector Dimension",
                        stats.get(
                            "dimension",
                            0
                        ),
                    )

                with c3:

                    st.metric(
                        "Documents",
                        len(
                            stats.get(
                                "sources",
                                []
                            )
                        ),
                    )

                st.write(
                    f"**Index Type:** "
                    f"{stats.get('index_type', 'Unknown')}"
                )

                st.write(
                    f"**Embedding Model:** "
                    f"`{EMBEDDING_MODEL}`"
                )

                st.write(
                    f"**LLM Model:** "
                    f"`{LLM_MODEL}`"
                )

                st.write(
                    f"**Vector DB Path:** "
                    f"`{VECTOR_DB_PATH}`"
                )

                saved_sources = stats.get(
                    "sources",
                    []
                )

                if saved_sources:

                    st.write(
                        "**Indexed Documents:**"
                    )

                    for source in saved_sources:

                        st.caption(
                            f"📄 {source}"
                        )

                st.divider()

                if st.button(
                    "🔄 Reload Vector Database",
                    use_container_width=True,
                ):

                    try:

                        db.load()

                        loaded_chunks = len(
                            getattr(
                                db,
                                "documents",
                                []
                            )
                        )

                        if loaded_chunks > 0:

                            st.session_state.db_created = True
                            st.session_state.db_loaded = True

                            st.success(
                                f"Vector Database Loaded: "
                                f"{loaded_chunks} chunks"
                            )

                            st.rerun()

                        else:

                            st.warning(
                                "Database contains no chunks."
                            )

                    except Exception as e:

                        st.error(
                            f"Database reload failed: {e}"
                        )


                if st.button(
                    "🗑 Clear Vector Database",
                    use_container_width=True,
                ):

                    try:

                        db.clear()

                        st.session_state.db_created = False
                        st.session_state.db_loaded = False
                        st.session_state.chunks = []
                        st.session_state.indexed_file_names = []
                        st.session_state.document_chunks = {}

                        st.success(
                            "Vector database cleared successfully."
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Failed to clear database: {e}"
                        )

            except Exception as e:

                st.error(
                    f"Database status error: {e}"
                )


        # ====================================================
        # SYSTEM CONFIGURATION
        # ====================================================

        with st.expander(
            "⚙ System Configuration",
            expanded=False,
        ):

            st.metric(
                "Default Chunk Size",
                CHUNK_SIZE,
            )

            st.metric(
                "Default Chunk Overlap",
                CHUNK_OVERLAP,
            )

            st.write(
                f"**Embedding Model:** "
                f"`{EMBEDDING_MODEL}`"
            )

            st.write(
                f"**LLM Model:** "
                f"`{LLM_MODEL}`"
            )

            st.write(
                f"**Vector DB Path:** "
                f"`{VECTOR_DB_PATH}`"
            )

            st.write(
                f"**Supported Files:** "
                f"{', '.join(SUPPORTED_FILES)}"
            )

    else:

        st.divider()

        st.subheader(
            "👤 User Access"
        )

        st.info(
            "You have standard user access."
        )


    # ========================================================
    # LOGOUT
    # ========================================================

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True,
    ):

        try:

            auth.logout()

        except Exception:

            pass

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.user_id = None
        st.session_state.full_name = ""
        st.session_state.email = ""

        st.session_state.chat_history = []

        st.session_state.all_documents = []
        st.session_state.combined_text = ""
        st.session_state.combined_tables = []
        st.session_state.chunks = []

        st.session_state.processed_file_names = []
        st.session_state.indexed_file_names = []
        st.session_state.document_chunks = {}

        st.session_state.db_loaded = False
        st.session_state.db_created = False

        st.rerun()


# ============================================================
# PERMANENT DOCUMENT LIBRARY
# ============================================================

st.header(
    "🗂 My Stored Documents"
)

st.info(
    "Your documents are permanently stored in "
    "Supabase Storage and remain available until "
    "you or an administrator deletes them."
)


try:

    stored_documents = (
        database.get_user_documents(
            st.session_state.user_id
        )
    )

    if not stored_documents:

        st.info(
            "No permanently stored documents yet."
        )

    else:

        st.success(
            f"📚 {len(stored_documents)} "
            f"stored document(s)"
        )

        for document in stored_documents:

            document_id = document.get(
                "id"
            )

            filename = document.get(
                "filename",
                "Unknown file"
            )

            file_size = document.get(
                "file_size",
                0
            )

            uploaded_at = document.get(
                "uploaded_at",
                "Unknown"
            )

            status = document.get(
                "status",
                "Unknown"
            )

            file_path = document.get(
                "file_path"
            )

            with st.expander(
                f"📄 {filename}"
            ):

                st.write(
                    f"**File:** {filename}"
                )

                st.write(
                    f"**Size:** {file_size:,} bytes"
                )

                st.write(
                    f"**Uploaded:** {uploaded_at}"
                )

                st.write(
                    f"**Status:** {status}"
                )

                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                if file_path:

                    if st.button(
                        "⬇️ Download",
                        key=f"download_{document_id}",
                        use_container_width=True,
                    ):

                        try:

                            file_data = (
                                database.supabase
                                .storage
                                .from_(
                                    database.storage_bucket
                                )
                                .download(
                                    file_path
                                )
                            )

                            st.download_button(
                                "📥 Save File",
                                data=file_data,
                                file_name=filename,
                                key=f"save_{document_id}",
                                use_container_width=True,
                            )

                        except Exception as e:

                            st.error(
                                f"Download failed: {e}"
                            )

                else:

                    st.warning(
                        "This document has no Supabase "
                        "Storage path. It may be an older "
                        "record created before permanent "
                        "storage was enabled."
                    )

                # ------------------------------------------------
                # DELETE
                # ------------------------------------------------

                can_delete = (
                    document.get(
                        "uploaded_by"
                    )
                    == st.session_state.user_id
                    or
                    st.session_state.role
                    == "Admin"
                )

                if can_delete:

                    if st.button(
                        "🗑 Delete Permanently",
                        key=f"delete_{document_id}",
                        use_container_width=True,
                    ):

                        try:

                            deleted = (
                                database.delete_document(
                                    document_id
                                )
                            )

                            if deleted:

                                st.success(
                                    f"{filename} "
                                    "deleted permanently."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "Document not found."
                                )

                        except Exception as e:

                            st.error(
                                f"Delete failed: {e}"
                            )


except Exception as e:

    st.error(
        f"Unable to load stored documents: {e}"
    )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header(
    "📂 Upload Enterprise Documents"
)

st.info(
    "Upload documents to permanently store them "
    "in your enterprise document library."
)

upload_types = [
    ext.replace(".", "")
    for ext in SUPPORTED_FILES
]

uploaded_files = st.file_uploader(
    "Upload Enterprise Documents",
    type=upload_types,
    accept_multiple_files=True,
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    current_file_names = sorted(
        file.name
        for file in uploaded_files
    )

    old_file_names = sorted(
        st.session_state.processed_file_names
    )

    documents_changed = (
        current_file_names != old_file_names
    )


    # --------------------------------------------------------
    # DOCUMENTS CHANGED
    # --------------------------------------------------------

    if documents_changed:

        # IMPORTANT:
        # This clears only the CURRENT VECTOR DATABASE.
        # It does NOT delete Supabase files.

        try:

            if db.exists():

                db.clear()

        except Exception as e:

            st.warning(
                f"Old vector database could not be cleared: {e}"
            )


        st.session_state.db_created = False
        st.session_state.db_loaded = False
        st.session_state.chunks = []
        st.session_state.indexed_file_names = []
        st.session_state.document_chunks = {}

        st.session_state.all_documents = []
        st.session_state.combined_text = ""
        st.session_state.combined_tables = []


    all_documents = []

    combined_text = ""

    combined_tables = []


    # ========================================================
    # EXTRACTION
    # ========================================================

    for uploaded_file in uploaded_files:

        original_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()


        if original_extension not in SUPPORTED_FILES:

            st.error(
                f"Unsupported file type: "
                f"{original_extension}"
            )

            continue


        temporary_file_path = None


        try:

            # ------------------------------------------------
            # READ FILE
            # ------------------------------------------------

            file_bytes = (
                uploaded_file.getvalue()
            )

            # ------------------------------------------------
            # PERMANENT SUPABASE PATH
            # ------------------------------------------------

            storage_path = make_storage_path(
                st.session_state.user_id,
                uploaded_file.name,
                file_bytes,
            )

            content_type = get_content_type(
                uploaded_file.name
            )

            # ------------------------------------------------
            # CHECK EXISTING USER DOCUMENTS
            # ------------------------------------------------

            existing_documents = (
                database.get_user_documents(
                    st.session_state.user_id
                )
            )

            existing_paths = {
                document.get("file_path")
                for document in existing_documents
                if document.get("file_path")
            }

            # ------------------------------------------------
            # PERMANENT STORAGE
            # ------------------------------------------------

            if storage_path not in existing_paths:

                database.upload_document_file(
                    file_bytes=file_bytes,
                    storage_path=storage_path,
                    content_type=content_type,
                )

                database.create_document(
                    filename=uploaded_file.name,
                    uploaded_by=st.session_state.user_id,
                    file_path=storage_path,
                    file_type=original_extension,
                    file_size=uploaded_file.size,
                    status="uploaded",
                )

                st.success(
                    f"💾 Permanently stored: "
                    f"{uploaded_file.name}"
                )

            else:

                st.info(
                    f"📁 Already stored: "
                    f"{uploaded_file.name}"
                )


            # ------------------------------------------------
            # TEMPORARY FILE FOR EXTRACTION
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=original_extension,
            ) as tmp:

                tmp.write(
                    file_bytes
                )

                temporary_file_path = (
                    tmp.name
                )


            # ------------------------------------------------
            # EXTRACTION
            # ------------------------------------------------

            extractor = DocumentExtractor(
                temporary_file_path
            )

            document = (
                extractor.extract_document()
            )


            if not isinstance(
                document,
                dict
            ):

                st.error(
                    f"Invalid extraction result "
                    f"for {uploaded_file.name}."
                )

                continue


            document.setdefault(
                "metadata",
                {}
            )


            document["metadata"][
                "uploaded_file_name"
            ] = uploaded_file.name


            document["metadata"][
                "storage_path"
            ] = storage_path


            document["source"] = (
                uploaded_file.name
            )


            all_documents.append(
                document
            )


            combined_text += (
                document.get(
                    "text",
                    ""
                )
                + "\n\n"
            )


            combined_tables.extend(
                document.get(
                    "tables",
                    []
                )
            )


        except Exception as e:

            st.error(
                f"Failed to process "
                f"{uploaded_file.name}: {e}"
            )


        finally:

            if temporary_file_path:

                try:

                    os.remove(
                        temporary_file_path
                    )

                except Exception:

                    pass


    # ========================================================
    # SAVE DOCUMENT STATE
    # ========================================================

    if all_documents:

        st.session_state.all_documents = (
            all_documents
        )

        st.session_state.combined_text = (
            combined_text
        )

        st.session_state.combined_tables = (
            combined_tables
        )

        st.session_state.processed_file_names = (
            current_file_names
        )


# ============================================================
# CURRENT DOCUMENT DATA
# ============================================================

all_documents = (
    st.session_state.all_documents
)

combined_text = (
    st.session_state.combined_text
)

combined_tables = (
    st.session_state.combined_tables
)


# ============================================================
# PROCESSING
# ============================================================

if all_documents:

    st.success(
        f"✅ {len(all_documents)} "
        "document(s) processed successfully."
    )


    # ========================================================
    # DOCUMENT STATISTICS
    # ========================================================

    st.header(
        "📊 Document Statistics"
    )


    total_extracted_images = sum(
        document.get(
            "statistics",
            {}
        ).get(
            "images",
            0
        )
        for document in all_documents
    )


    total_visuals = sum(
        document.get(
            "statistics",
            {}
        ).get(
            "visuals",
            0
        )
        for document in all_documents
    )


    total_pages = sum(
        int(
            document.get(
                "metadata",
                {}
            ).get(
                "pages",
                1
            )
        )
        for document in all_documents
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Documents",
        len(all_documents)
    )

    c2.metric(
        "Pages",
        total_pages
    )

    c3.metric(
        "Tables",
        len(combined_tables)
    )

    c4.metric(
        "Images",
        total_extracted_images
    )

    c5.metric(
        "Visuals",
        total_visuals
    )


    # ========================================================
    # DOCUMENT DETAILS
    # ========================================================

    st.header(
        "📄 Document Details"
    )


    for doc in all_documents:

        meta = doc.get(
            "metadata",
            {}
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document"
            )
        )


        with st.expander(
            f"📄 {display_name}"
        ):

            st.write(meta)


    # ========================================================
    # EXTRACTED IMAGES
    # ========================================================

    st.header(
        "🖼 Extracted Images"
    )


    images_found = False


    for doc in all_documents:

        meta = doc.get(
            "metadata",
            {}
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document"
            )
        )


        for page_data in doc.get(
            "images",
            []
        ):

            if not isinstance(
                page_data,
                dict
            ):
                continue


            image_list = page_data.get(
                "images",
                []
            )


            valid_images = [
                image
                for image in image_list
                if (
                    isinstance(
                        image,
                        dict
                    )
                    and image.get("path")
                    and os.path.exists(
                        image.get("path")
                    )
                )
            ]


            if not valid_images:

                continue


            images_found = True


            page_number = page_data.get(
                "page",
                1
            )


            st.subheader(
                f"📄 {display_name} | "
                f"Page {page_number}"
            )


            image_columns = st.columns(
                min(
                    3,
                    len(valid_images)
                )
            )


            for index, image in enumerate(
                valid_images
            ):

                with image_columns[
                    index % len(image_columns)
                ]:

                    st.image(
                        image["path"],
                        caption=(
                            f"Image "
                            f"{image.get('image_no', '')}"
                            f" | Page "
                            f"{page_number}"
                        ),
                        use_container_width=True,
                    )


    if not images_found:

        st.info(
            "No embedded raster images were detected."
        )


    # ========================================================
    # VISUALS
    # ========================================================

    st.header(
        "📊 Visuals, Flowcharts & Diagrams"
    )


    visuals_found = False


    for doc in all_documents:

        meta = doc.get(
            "metadata",
            {}
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document"
            )
        )


        for visual in doc.get(
            "visuals",
            []
        ):

            if not isinstance(
                visual,
                dict
            ):
                continue


            visual_path = visual.get(
                "path"
            )


            if (
                not visual_path
                or
                not os.path.exists(
                    visual_path
                )
            ):

                continue


            visuals_found = True


            page_number = visual.get(
                "page",
                "?"
            )


            with st.expander(
                f"📑 {display_name} | "
                f"Page {page_number} | PDF Visual"
            ):

                st.image(
                    visual_path,
                    use_container_width=True,
                )


    if not visuals_found:

        st.info(
            "No PDF visual pages, flowcharts, "
            "or diagrams were detected."
        )


    # ========================================================
    # TABLES
    # ========================================================

    st.header(
        "📋 Extracted Tables"
    )


    if combined_tables:

        for table_index, table in enumerate(
            combined_tables,
            start=1
        ):

            if "table" in table:

                location = table.get(
                    "page",
                    table.get(
                        "sheet",
                        table_index
                    )
                )


                st.subheader(
                    f"Table {table_index} | "
                    f"Page/Sheet {location}"
                )


                df = pd.DataFrame(
                    table["table"]
                )


                st.dataframe(
                    df,
                    use_container_width=True
                )


            elif "error" in table:

                st.error(
                    table["error"]
                )


    else:

        st.info(
            "No tables were detected."
        )


    # ========================================================
    # CHUNKING
    # ========================================================

    engine = ChunkingEngine(

        text=combined_text,

        metadata={
            "documents": len(all_documents),
            "pages": total_pages,
        },

        tables=combined_tables,

        source=", ".join(
            [
                d.get(
                    "metadata",
                    {}
                ).get(
                    "uploaded_file_name",
                    "Unknown"
                )
                for d in all_documents
            ]
        ),

        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


    st.header(
        "✂ Chunking"
    )


    st.info(
        f"**Strategy:** {chunk_method} | "
        f"**Chunk Size:** {chunk_size} | "
        f"**Chunk Overlap:** {chunk_overlap}"
    )


    if chunk_method == "Character":

        chunks = engine.character_chunking()

    elif chunk_method == "Recursive":

        chunks = engine.recursive_chunking()

    elif chunk_method == "Token":

        chunks = engine.token_chunking()

    elif chunk_method == "Markdown":

        chunks = engine.markdown_chunking()

    elif chunk_method == "Context":

        chunks = engine.contextual_chunking()

    elif chunk_method == "Table":

        chunks = engine.table_chunking()

    else:

        chunks = []


    if chunks is None:

        chunks = []


    # ========================================================
    # SOURCE IN EVERY CHUNK
    # ========================================================

    current_sources = [
        d.get(
            "metadata",
            {}
        ).get(
            "uploaded_file_name",
            "Unknown"
        )
        for d in all_documents
    ]


    for chunk in chunks:

        if not isinstance(
            chunk,
            dict
        ):

            continue


        if not chunk.get(
            "source"
        ):

            chunk["source"] = ", ".join(
                current_sources
            )


    st.session_state.chunks = chunks


    st.success(
        f"✅ {len(chunks)} chunks generated "
        f"using {chunk_method} chunking."
    )


    # ========================================================
    # VECTOR DATABASE
    # ========================================================

    st.header(
        "🧠 Vector Database"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "Create / Replace Vector Database",
            use_container_width=True,
        ):

            try:

                if not chunks:

                    st.error(
                        "No chunks available."
                    )

                else:

                    db.clear()

                    db.create_index(
                        chunks
                    )

                    db.save()

                    db.load()

                    loaded_chunks = len(
                        getattr(
                            db,
                            "documents",
                            []
                        )
                    )

                    saved_sources = (
                        db.get_saved_sources()
                    )

                    if (
                        loaded_chunks > 0
                        and
                        sorted(saved_sources)
                        == sorted(current_sources)
                    ):

                        st.session_state.db_created = True
                        st.session_state.db_loaded = True

                        st.session_state.indexed_file_names = (
                            current_file_names
                        )

                        st.success(
                            "✅ Vector Database "
                            "replaced successfully.\n\n"
                            f"📄 Documents: "
                            f"{len(current_sources)}\n\n"
                            f"🧩 Chunks: "
                            f"{loaded_chunks}"
                        )

                        st.rerun()

                    else:

                        st.session_state.db_created = False
                        st.session_state.db_loaded = False

                        st.error(
                            "Vector Database was created "
                            "but source validation failed."
                        )

            except Exception as e:

                st.session_state.db_created = False
                st.session_state.db_loaded = False

                st.error(
                    f"Vector database creation failed: {e}"
                )


    # ========================================================
    # DATABASE STATUS
    # ========================================================

    with col2:

        if st.session_state.db_loaded:

            st.success(
                "🟢 Database Ready"
            )

        else:

            st.warning(
                "🟡 Database Not Ready"
            )


    # ========================================================
    # LOADED CHUNKS
    # ========================================================

    with col3:

        st.metric(
            "Loaded Chunks",
            len(
                getattr(
                    db,
                    "documents",
                    []
                )
            )
            if st.session_state.db_loaded
            else 0,
        )


    # ========================================================
    # AGENTIC RAG
    # ========================================================

    st.header(
        "🤖 Agentic RAG"
    )


    loaded_count = len(
        getattr(
            db,
            "documents",
            []
        )
    )


    actual_db_ready = (
        getattr(
            db,
            "index",
            None
        ) is not None
        and
        loaded_count > 0
    )


    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    database_matches_current_documents = False


    if actual_db_ready:

        database_matches_current_documents = (
            db.matches_sources(
                current_sources
            )
        )


    if (
        actual_db_ready
        and
        database_matches_current_documents
    ):

        st.session_state.db_created = True
        st.session_state.db_loaded = True

        saved_sources = (
            db.get_saved_sources()
        )

        source_text = "\n".join(
            "- " + source
            for source in saved_sources
        )

        st.success(
            f"""
**EnterpriseRAGAgent is active.**

🟢 Vector database matches the currently uploaded documents.

**Indexed chunks:** {loaded_count}

**Indexed documents:**

{source_text}
"""
        )


    elif (
        actual_db_ready
        and
        not database_matches_current_documents
    ):

        st.session_state.db_loaded = False

        st.warning(
            """
⚠️ The existing vector database belongs to different
documents.

Please click **Create / Replace Vector Database**
before asking questions.
"""
        )


    else:

        st.session_state.db_loaded = False

        st.warning(
            """
Vector database is not loaded.

Please click **Create / Replace Vector Database**
before asking questions.
"""
        )


    # ========================================================
    # ENTERPRISE CHAT
    # ========================================================

    st.header(
        "💬 Enterprise Chat"
    )


    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    query = st.chat_input(
        "Ask anything about your documents..."
    )


    if query:

        with st.chat_message(
            "user"
        ):

            st.markdown(
                query
            )


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query,
            }
        )


        try:

            if not actual_db_ready:

                st.warning(
                    "Please create the Vector Database first."
                )


            elif not database_matches_current_documents:

                st.warning(
                    "The vector database does not match "
                    "the currently uploaded documents. "
                    "Please click Create / Replace "
                    "Vector Database."
                )


            else:

                llm = GeminiLLM()

                agent = EnterpriseRAGAgent(
                    vector_db=db,
                    llm=llm,
                    documents=all_documents,
                    top_k=top_k,
                )

                with st.spinner(
                    "🤖 Agent is deciding how to answer..."
                ):

                    response = agent.run(
                        query
                    )


                if not isinstance(
                    response,
                    dict
                ):

                    raise ValueError(
                        "EnterpriseRAGAgent returned "
                        "an invalid response."
                    )


                answer = response.get(
                    "answer",
                    "No answer generated."
                )

                results = response.get(
                    "sources",
                    []
                )

                action = response.get(
                    "action",
                    "unknown"
                )

                iterations = response.get(
                    "iterations",
                    0
                )

                trace = response.get(
                    "trace",
                    []
                )


                with st.chat_message(
                    "assistant"
                ):

                    st.markdown(
                        answer
                    )

                    st.caption(
                        f"🤖 Agent Action: `{action}` "
                        f"| Iterations: `{iterations}`"
                    )


                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


                st.subheader(
                    "🤖 Agent Execution Trace"
                )


                trace_col1, trace_col2, trace_col3 = (
                    st.columns(3)
                )


                with trace_col1:

                    st.metric(
                        "Agent Decision",
                        action
                    )


                with trace_col2:

                    st.metric(
                        "Retrieved Sources",
                        len(results)
                    )


                with trace_col3:

                    st.metric(
                        "Iterations",
                        iterations
                    )


                if trace:

                    with st.expander(
                        "🔎 View Detailed Agent Trace"
                    ):

                        for trace_item in trace:

                            st.write(
                                trace_item
                            )


                st.subheader(
                    "📚 Sources Used"
                )


                if not results:

                    st.warning(
                        "No relevant sources found."
                    )

                else:

                    current_source_set = set(
                        current_sources
                    )

                    for i, result in enumerate(
                        results
                    ):

                        if not isinstance(
                            result,
                            dict
                        ):

                            continue


                        result_source = str(
                            result.get(
                                "source",
                                ""
                            )
                        )


                        if (
                            result_source
                            and
                            result_source
                            not in current_source_set
                        ):

                            continue


                        with st.expander(
                            f"Source {i + 1}"
                        ):

                            st.write(
                                f"**Chunk ID:** "
                                f"{result.get('chunk_id', 'N/A')}"
                            )

                            st.write(
                                f"**Chunk Type:** "
                                f"{result.get('chunk_type', 'Unknown')}"
                            )

                            if "page" in result:

                                st.write(
                                    f"**Page:** "
                                    f"{result.get('page')}"
                                )

                            st.write(
                                f"**Source:** "
                                f"{result.get('source', 'Unknown')}"
                            )

                            if "similarity_score" in result:

                                st.write(
                                    "**Similarity / Match Score:** "
                                    f"{result.get('similarity_score')}"
                                )

                            content = result.get(
                                "content",
                                ""
                            )

                            if isinstance(
                                content,
                                str
                            ):

                                st.markdown(
                                    content
                                )

                            else:

                                st.write(
                                    content
                                )


        except Exception as e:

            st.error(
                f"Agentic RAG failed:\n\n{e}"
            )


# ============================================================
# NO DOCUMENTS
# ============================================================

else:

    st.info(
        "📂 Upload one or more enterprise documents "
        "to begin processing."
    )


# ============================================================
# CONVERSATION TOOLS
# ============================================================

st.divider()


col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


conversation = ""


for message in st.session_state.chat_history:

    conversation += (
        f"{message['role'].upper()}:\n"
    )

    conversation += str(
        message["content"]
    )

    conversation += "\n\n"


with col2:

    st.download_button(
        "📥 Download Chat",
        conversation,
        file_name="conversation.txt",
        mime="text/plain",
        use_container_width=True,
    )


with col3:

    if st.button(
        "📝 Summarize Chat",
        use_container_width=True,
    ):

        if not st.session_state.chat_history:

            st.warning(
                "No conversation found."
            )

        else:

            history = ""

            for msg in st.session_state.chat_history:

                history += (
                    f"{msg['role']}: "
                    f"{msg['content']}\n"
                )

            try:

                llm = GeminiLLM()

                summary = llm.generate_answer(
                    query=(
                        "Summarize this "
                        "conversation in "
                        "bullet points."
                    ),
                    chunks=[
                        {
                            "content": history
                        }
                    ],
                )

                st.subheader(
                    "📋 Conversation Summary"
                )

                st.write(
                    summary
                )

            except Exception as e:

                st.error(
                    f"Summary failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise Document Intelligence System | "
    "Phase 4 | Agentic RAG Architecture | "
    "Permanent Supabase Document Storage"
)