import os
import json
import tempfile

import pandas as pd
import streamlit as st

from extractor import DocumentExtractor
from chunking import ChunkingEngine
from vector_db import VectorDatabase
from llm import GeminiLLM
from auth import Authentication
from register import Register
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
# PAGE CONFIGURATION
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
    "chat_history": [],
    "db_created": False,
    "db_loaded": False,
    "all_documents": [],
    "combined_text": "",
    "combined_tables": [],
    "chunks": [],
    "processed_file_names": [],
}


for key, default_value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# AUTHENTICATION OBJECTS
# ============================================================

auth = Authentication()
register = Register()


# ============================================================
# AUTHENTICATION SCREEN
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

        st.session_state.vector_db = VectorDatabase()

    except Exception as e:

        st.error(
            f"Failed to initialize Vector Database: {e}"
        )

        st.stop()


db = st.session_state.vector_db


# ============================================================
# AUTO LOAD EXISTING DATABASE
# ============================================================

if not st.session_state.db_loaded:

    try:

        if db.exists():

            db.load()

            if (
                getattr(
                    db,
                    "index",
                    None
                ) is not None
                and len(
                    getattr(
                        db,
                        "documents",
                        []
                    )
                ) > 0
            ):

                st.session_state.db_created = True
                st.session_state.db_loaded = True

        else:

            st.session_state.db_created = False
            st.session_state.db_loaded = False

    except Exception as e:

        st.session_state.db_created = False
        st.session_state.db_loaded = False

        st.warning(
            "Vector database exists but could not be loaded: "
            f"{e}"
        )


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

    st.divider()

    st.header("⚙ Settings")

    # ========================================================
    # CHUNKING CONTROLS
    # ========================================================

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

        st.info(
            "You have administrator access."
        )

        # ====================================================
        # USER MANAGEMENT
        # ====================================================

        with st.expander("👥 User Management"):

            st.write("Registered Users")

            try:

                with open(
                    "users.json",
                    "r",
                    encoding="utf-8",
                ) as f:

                    users = json.load(f)

                if not users:

                    st.warning(
                        "No registered users found."
                    )

                else:

                    for user in users:

                        st.markdown("---")

                        st.write(
                            f"**Username:** "
                            f"{user.get('username', 'Unknown')}"
                        )

                        st.write(
                            f"**Name:** "
                            f"{user.get('full_name', 'Unknown')}"
                        )

                        st.write(
                            f"**Email:** "
                            f"{user.get('email', 'Unknown')}"
                        )

                        st.write(
                            f"**Role:** "
                            f"{user.get('role', 'User')}"
                        )

            except FileNotFoundError:

                st.warning(
                    "users.json was not found."
                )

            except Exception as e:

                st.error(
                    f"Unable to load users: {e}"
                )

        # ====================================================
        # SYSTEM CONFIGURATION
        # ====================================================

        with st.expander("⚙ System Configuration"):

            st.write(
                "Current system configuration"
            )

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

            supported = ", ".join(
                SUPPORTED_FILES
            )

            st.write(
                f"**Supported Files:** "
                f"{supported}"
            )

        # ====================================================
        # DATABASE CONTROLS
        # ====================================================

        with st.expander("🗄 Database Controls"):

            try:

                stats = db.get_stats()

                status = stats.get(
                    "status",
                    "Unknown",
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

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Indexed Chunks",
                        stats.get(
                            "chunks",
                            0,
                        ),
                    )

                with c2:

                    st.metric(
                        "Vector Dimension",
                        stats.get(
                            "dimension",
                            0,
                        ),
                    )

                st.write(
                    f"**Index Type:** "
                    f"{stats.get('index_type', 'Unknown')}"
                )

                st.write(
                    f"**Database Path:** "
                    f"`{stats.get('path', VECTOR_DB_PATH)}`"
                )

                st.divider()

                # ============================================
                # RELOAD DATABASE
                # ============================================

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
                                [],
                            )
                        )

                        if loaded_chunks > 0:

                            st.session_state.db_created = True
                            st.session_state.db_loaded = True

                            st.success(
                                "Vector Database Loaded: "
                                f"{loaded_chunks} chunks"
                            )

                        else:

                            st.session_state.db_created = False
                            st.session_state.db_loaded = False

                            st.warning(
                                "Vector Database loaded, "
                                "but contains no chunks."
                            )

                        st.rerun()

                    except Exception as e:

                        st.session_state.db_loaded = False

                        st.error(
                            f"Database reload failed: {e}"
                        )

                # ============================================
                # CLEAR DATABASE
                # ============================================

                if st.button(
                    "🗑 Clear Vector Database",
                    use_container_width=True,
                ):

                    try:

                        db.clear()

                        st.session_state.db_created = False
                        st.session_state.db_loaded = False
                        st.session_state.chunks = []

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

    else:

        st.divider()

        st.subheader("👤 User Access")

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

        st.session_state.chat_history = []
        st.session_state.db_loaded = False
        st.session_state.db_created = False

        st.session_state.all_documents = []
        st.session_state.combined_text = ""
        st.session_state.combined_tables = []
        st.session_state.chunks = []
        st.session_state.processed_file_names = []

        st.rerun()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header(
    "📂 Upload Enterprise Documents"
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
# DOCUMENT VARIABLES
# ============================================================

all_documents = []
combined_text = ""
combined_tables = []


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_files:

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

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=original_extension,
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temporary_file_path = tmp.name

            extractor = DocumentExtractor(
                temporary_file_path
            )

            document = extractor.extract_document()

            if not isinstance(
                document,
                dict,
            ):

                st.error(
                    f"Invalid extraction result "
                    f"for {uploaded_file.name}."
                )

                continue

            document.setdefault(
                "metadata",
                {},
            )

            document["metadata"][
                "uploaded_file_name"
            ] = uploaded_file.name

            all_documents.append(
                document
            )

            combined_text += (
                document.get(
                    "text",
                    "",
                )
                + "\n\n"
            )

            combined_tables.extend(
                document.get(
                    "tables",
                    [],
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
    # SAVE DOCUMENTS IN SESSION STATE
    # ========================================================

    if all_documents:

        st.session_state.all_documents = all_documents

        st.session_state.combined_text = combined_text

        st.session_state.combined_tables = combined_tables

        st.session_state.processed_file_names = [
            file.name
            for file in uploaded_files
        ]


# ============================================================
# LOAD SESSION DOCUMENT DATA
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
# PROCESSING SUCCESS
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
            {},
        ).get(
            "images",
            0,
        )
        for document in all_documents
    )

    total_visuals = sum(
        document.get(
            "statistics",
            {},
        ).get(
            "visuals",
            0,
        )
        for document in all_documents
    )

    total_pages = sum(
        int(
            document.get(
                "metadata",
                {},
            ).get(
                "pages",
                1,
            )
        )
        for document in all_documents
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Documents",
        len(all_documents),
    )

    c2.metric(
        "Pages",
        total_pages,
    )

    c3.metric(
        "Tables",
        len(combined_tables),
    )

    c4.metric(
        "Images",
        total_extracted_images,
    )

    c5.metric(
        "Visuals",
        total_visuals,
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
            {},
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document",
            ),
        )

        with st.expander(
            display_name
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
            {},
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document",
            ),
        )

        document_images = doc.get(
            "images",
            [],
        )

        for page_data in document_images:

            if not isinstance(
                page_data,
                dict,
            ):

                continue

            image_list = page_data.get(
                "images",
                [],
            )

            if not image_list:
                continue

            valid_images = [
                image
                for image in image_list
                if (
                    isinstance(
                        image,
                        dict,
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
                1,
            )

            st.subheader(
                f"📄 {display_name} "
                f"| Page {page_number}"
            )

            image_columns = st.columns(
                min(
                    3,
                    len(valid_images),
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

                    width = image.get(
                        "width"
                    )

                    height = image.get(
                        "height"
                    )

                    if width and height:

                        st.caption(
                            f"{width} × "
                            f"{height} px"
                        )

    if not images_found:

        st.info(
            "No embedded raster images "
            "were detected."
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
            {},
        )

        display_name = meta.get(
            "uploaded_file_name",
            meta.get(
                "file_name",
                "Document",
            ),
        )

        document_visuals = doc.get(
            "visuals",
            [],
        )

        for visual in document_visuals:

            if not isinstance(
                visual,
                dict,
            ):

                continue

            visual_path = visual.get(
                "path"
            )

            if not visual_path:
                continue

            if not os.path.exists(
                visual_path
            ):

                continue

            visuals_found = True

            page_number = visual.get(
                "page",
                "?",
            )

            with st.expander(
                f"📑 {display_name} "
                f"| Page {page_number} "
                "| PDF Visual"
            ):

                st.image(
                    visual_path,
                    caption=(
                        f"{display_name} "
                        f"| Page {page_number}"
                    ),
                    use_container_width=True,
                )

                embedded_images = visual.get(
                    "embedded_images",
                    0,
                )

                drawings = visual.get(
                    "drawings",
                    0,
                )

                st.caption(
                    f"Embedded images: "
                    f"{embedded_images} "
                    f"| Vector drawing objects: "
                    f"{drawings}"
                )

    if not visuals_found:

        st.info(
            "No PDF visual pages, "
            "flowcharts, or diagrams "
            "were detected."
        )

    # ========================================================
    # TABLE EXTRACTION
    # ========================================================

    st.header(
        "📋 Extracted Tables"
    )

    if combined_tables:

        for table_index, table in enumerate(
            combined_tables,
            start=1,
        ):

            if "table" in table:

                location = table.get(
                    "page",
                    table.get(
                        "sheet",
                        table_index,
                    ),
                )

                st.subheader(
                    f"Table {table_index} "
                    f"| Page/Sheet {location}"
                )

                df = pd.DataFrame(
                    table["table"]
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                )

            elif "error" in table:

                st.error(
                    table["error"]
                )

    else:

        st.info(
            "No tables were detected in "
            "the uploaded documents."
        )

    # ========================================================
    # CHUNKING ENGINE
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
                    {},
                ).get(
                    "uploaded_file_name",
                    d.get(
                        "metadata",
                        {},
                    ).get(
                        "file_name",
                        "Unknown",
                    ),
                )
                for d in all_documents
            ]
        ),

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,
    )

    # ========================================================
    # CHUNKING
    # ========================================================

    st.header(
        "✂ Chunking"
    )

    st.info(
        f"**Strategy:** {chunk_method}  |  "
        f"**Chunk Size:** {chunk_size}  |  "
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

    st.session_state.chunks = chunks

    st.success(
        f"✅ {len(chunks)} chunks generated "
        f"using {chunk_method} chunking."
    )

    # ========================================================
    # CHUNK STATISTICS
    # ========================================================

    stats = engine.chunk_statistics(
        chunks
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Chunks",
        stats["Total Chunks"],
    )

    c2.metric(
        "Average Length",
        stats["Average Length"],
    )

    c3.metric(
        "Maximum Length",
        stats["Maximum Length"],
    )

    c4.metric(
        "Minimum Length",
        stats["Minimum Length"],
    )

    c5.metric(
        "Configured Size",
        chunk_size,
    )

    # ========================================================
    # CHUNK PREVIEW
    # ========================================================

    st.header(
        "📄 Chunk Preview"
    )

    if chunks:

        preview_max = min(
            10,
            len(chunks),
        )

        preview_default = min(
            3,
            len(chunks),
        )

        preview = st.slider(
            "Preview Chunks",
            min_value=1,
            max_value=preview_max,
            value=preview_default,
        )

        for chunk in chunks[:preview]:

            with st.expander(
                f"{chunk.get('chunk_type', 'Chunk')} "
                f"| Chunk {chunk.get('chunk_id', '')}"
            ):

                st.write(
                    chunk.get(
                        "content",
                        "",
                    )
                )

                if "page" in chunk:

                    st.caption(
                        f"Page: {chunk['page']}"
                    )

    else:

        st.warning(
            "No chunks were generated."
        )

    # ========================================================
    # VECTOR DATABASE
    # ========================================================

    st.header(
        "🧠 Vector Database"
    )

    col1, col2, col3 = st.columns(3)

    # ========================================================
    # CREATE DATABASE
    # ========================================================

    with col1:

        if st.button(
            "Create Vector Database",
            use_container_width=True,
        ):

            try:

                if not chunks:

                    st.error(
                        "No chunks available. "
                        "Please process documents first."
                    )

                else:

                    db.create_index(
                        chunks
                    )

                    db.save()

                    db.load()

                    loaded_chunks = len(
                        getattr(
                            db,
                            "documents",
                            [],
                        )
                    )

                    if loaded_chunks > 0:

                        st.session_state.db_created = True
                        st.session_state.db_loaded = True

                        st.success(
                            "✅ Vector Database Created "
                            f"and Loaded "
                            f"({loaded_chunks} chunks)"
                        )

                        st.rerun()

                    else:

                        st.session_state.db_created = False
                        st.session_state.db_loaded = False

                        st.error(
                            "Database was created but "
                            "contains no chunks."
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

        if st.session_state.db_loaded:

            st.metric(
                "Loaded Chunks",
                len(
                    getattr(
                        db,
                        "documents",
                        [],
                    )
                ),
            )

        else:

            st.metric(
                "Loaded Chunks",
                0,
            )

    # ========================================================
    # AGENTIC RAG STATUS
    # ========================================================

    st.header(
        "🤖 Agentic RAG"
    )

    loaded_count = len(
        getattr(
            db,
            "documents",
            [],
        )
    )

    actual_db_ready = (
        getattr(
            db,
            "index",
            None
        ) is not None
        and loaded_count > 0
    )

    if actual_db_ready:

        st.session_state.db_created = True
        st.session_state.db_loaded = True

        st.success(
            f"""
**EnterpriseRAGAgent is active.**

🟢 Vector database loaded successfully.

**Indexed chunks:** {loaded_count}

The agent can autonomously:

- Decide the retrieval strategy
- Search the vector database
- Search document metadata
- Search tables
- Evaluate retrieved results
- Refine the query when results are insufficient
- Generate the final answer
"""
        )

    else:

        st.session_state.db_loaded = False

        st.warning(
            """
Vector database is not loaded.

Please click **Create Vector Database** above
or use **Reload Vector Database** from the Admin
Database Controls.
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

    # ========================================================
    # AGENTIC SEARCH + AI ANSWER
    # ========================================================

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

            # =================================================
            # DATABASE CHECK
            # =================================================

            if not actual_db_ready:

                st.warning(
                    "Please create or reload the "
                    "Vector Database first."
                )

            else:

                # =============================================
                # CREATE LLM
                # =============================================

                llm = GeminiLLM()

                # =============================================
                # CREATE AGENT
                # =============================================

                agent = EnterpriseRAGAgent(
                    vector_db=db,
                    llm=llm,
                    documents=all_documents,
                    top_k=top_k,
                )

                # =============================================
                # AGENT EXECUTION
                # =============================================

                with st.spinner(
                    "🤖 Agent is deciding how to answer..."
                ):

                    response = agent.run(
                        query
                    )

                if not isinstance(
                    response,
                    dict,
                ):

                    raise ValueError(
                        "EnterpriseRAGAgent returned "
                        "an invalid response."
                    )

                # =============================================
                # RESPONSE CONTRACT
                # =============================================

                answer = response.get(
                    "answer",
                    "No answer generated.",
                )

                results = response.get(
                    "sources",
                    [],
                )

                action = response.get(
                    "action",
                    "unknown",
                )

                success = response.get(
                    "success",
                    False,
                )

                iterations = response.get(
                    "iterations",
                    0,
                )

                trace = response.get(
                    "trace",
                    [],
                )

                # =============================================
                # ANSWER
                # =============================================

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

                # =============================================
                # AGENT EXECUTION TRACE
                # =============================================

                st.subheader(
                    "🤖 Agent Execution Trace"
                )

                trace_col1, trace_col2, trace_col3 = (
                    st.columns(3)
                )

                with trace_col1:

                    st.metric(
                        "Agent Decision",
                        action,
                    )

                with trace_col2:

                    st.metric(
                        "Retrieved Sources",
                        len(results),
                    )

                with trace_col3:

                    st.metric(
                        "Iterations",
                        iterations,
                    )

                if action == "table_search":

                    st.success(
                        "🟢 Agent selected TABLE SEARCH"
                    )

                elif action == "document_search":

                    st.success(
                        "🟢 Agent selected DOCUMENT SEARCH"
                    )

                elif action == "vector_search":

                    st.success(
                        "🟢 Agent selected VECTOR SEARCH"
                    )

                else:

                    st.info(
                        f"Agent selected: {action}"
                    )

                if not success:

                    st.warning(
                        "Agent could not generate "
                        "a successful retrieval response."
                    )

                # =============================================
                # TRACE DETAILS
                # =============================================

                if trace:

                    with st.expander(
                        "🔎 View Detailed Agent Trace"
                    ):

                        for trace_item in trace:

                            st.write(
                                trace_item
                            )

                # =============================================
                # SOURCES
                # =============================================

                st.subheader(
                    "📚 Sources Used"
                )

                if not results:

                    st.warning(
                        "No relevant sources found."
                    )

                else:

                    for i, result in enumerate(
                        results
                    ):

                        if not isinstance(
                            result,
                            dict,
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
                                "",
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


# ============================================================
# CLEAR CONVERSATION
# ============================================================

with col1:

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# DOWNLOAD CHAT
# ============================================================

conversation = ""

for message in st.session_state.chat_history:

    conversation += (
        f"{message['role'].upper()}:\n"
    )

    conversation += (
        str(
            message["content"]
        )
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


# ============================================================
# SUMMARIZE CHAT
# ============================================================

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
    "Phase 4 | Agentic RAG Architecture"
)