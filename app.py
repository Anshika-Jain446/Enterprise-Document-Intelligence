import streamlit as st
import tempfile
import os
import pandas as pd

from extractor import DocumentExtractor
from chunking import ChunkingEngine
from vector_db import VectorDatabase

st.set_page_config(
    page_title="Enterprise Document Intelligence System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Enterprise Document Intelligence System")
st.markdown("### Enterprise RAG Preprocessing Pipeline")

uploaded_files = st.file_uploader(
    "Upload Company Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_documents = []
    temp_files = []

    combined_text = ""
    combined_tables = []

    total_pages = 0
    total_words = 0
    total_characters = 0
    total_lines = 0

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
            temp_files.append(pdf_path)

        extractor = DocumentExtractor(pdf_path)

        document = extractor.extract_document()

        all_documents.append(document)

        combined_text += document.get(
            "text",
            ""
        ) + "\n\n"

        combined_tables.extend(
            document.get(
                "tables",
                []
            )
        )

        stats = document.get(
            "statistics",
            {}
        )

        total_words += stats.get(
            "words",
            0
        )

        total_characters += stats.get(
            "characters",
            0
        )

        total_lines += stats.get(
            "lines",
            0
        )

        total_pages += document.get(
            "metadata",
            {}
        ).get(
            "pages",
            0
        )

    st.success(
        f"✅ {len(all_documents)} documents processed successfully."
    )

    # -----------------------------------
    # Document Summary
    # -----------------------------------

    st.header("📊 Enterprise Document Summary")

    c1, c2, c3, c4 = st.columns(4)

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
        "Words",
        total_words
    )

    # -----------------------------------
    # Document Details
    # -----------------------------------

    st.header("📄 Individual Documents")

    for doc in all_documents:

        meta = doc.get(
            "metadata",
            {}
        )

        with st.expander(
            meta.get(
                "file_name",
                "Unknown Document"
            )
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Title:**",
                    meta.get("title")
                )

                st.write(
                    "**Author:**",
                    meta.get("author")
                )

                st.write(
                    "**Pages:**",
                    meta.get("pages")
                )

            with col2:

                st.write(
                    "**Subject:**",
                    meta.get("subject")
                )

                st.write(
                    "**Creator:**",
                    meta.get("creator")
                )

                st.write(
                    "**Producer:**",
                    meta.get("producer")
                )

    # -----------------------------------
    # Tables
    # -----------------------------------

    st.header("📋 Extracted Tables")

    if len(combined_tables) == 0:

        st.info("No tables detected.")

    else:

        for table in combined_tables:

            st.subheader(
                f"Page {table.get('page','Unknown')}"
            )

            df = pd.DataFrame(
                table.get(
                    "table",
                    []
                )
            )

            st.dataframe(
                df,
                use_container_width=True
            )

    # -----------------------------------
    # Chunking
    # -----------------------------------

    st.header("✂️ Chunking")

    method = st.selectbox(

        "Select Chunking Method",

        [
            "Character",
            "Recursive",
            "Token",
            "Markdown",
            "Context",
            "Table"
        ]

    )

    engine = ChunkingEngine(

        text=combined_text,

        metadata={

            "documents": len(all_documents),

            "pages": total_pages

        },

        tables=combined_tables,

        source="Company Knowledge Base"

    )

    if method == "Character":

        chunks = engine.character_chunking()

    elif method == "Recursive":

        chunks = engine.recursive_chunking()

    elif method == "Token":

        chunks = engine.token_chunking()

    elif method == "Markdown":

        chunks = engine.markdown_chunking()

    elif method == "Context":

        chunks = engine.contextual_chunking()

    else:

        chunks = engine.table_chunking()

    chunk_stats = engine.chunk_statistics(
        chunks
    )

    # -----------------------------------
    # Chunk Statistics
    # -----------------------------------

    st.header("📈 Chunk Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Chunks",
        chunk_stats.get(
            "Total Chunks",
            0
        )
    )

    c2.metric(
        "Average Length",
        chunk_stats.get(
            "Average Length",
            0
        )
    )

    c3.metric(
        "Maximum Length",
        chunk_stats.get(
            "Maximum Length",
            0
        )
    )

    c4.metric(
        "Minimum Length",
        chunk_stats.get(
            "Minimum Length",
            0
        )
    )

        # -----------------------------------
    # Chunk Preview
    # -----------------------------------

    st.header("📄 Chunk Preview")

    if len(chunks) > 0:

        preview = st.slider(
            "Preview First N Chunks",
            1,
            min(10, len(chunks)),
            min(3, len(chunks))
        )

        for chunk in chunks[:preview]:

            with st.expander(
                f"{chunk.get('chunk_type','Unknown')} - Chunk {chunk.get('chunk_id','')}"
            ):

                st.write(
                    chunk.get(
                        "content",
                        ""
                    )
                )

                if "page" in chunk:

                    st.caption(
                        f"Page : {chunk['page']}"
                    )

    else:

        st.warning(
            "No chunks generated."
        )

    # -----------------------------------
    # Vector Database
    # -----------------------------------

    st.header("🧠 Vector Database")

    if st.button("Create Vector Database"):

        if len(chunks) == 0:

            st.warning(
                "No chunks available."
            )

        else:

            try:

                db = VectorDatabase()

                db.create_index(
                    chunks
                )

                db.save()

                st.success(
                    "✅ Vector Database Created Successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error creating Vector Database:\n\n{e}"
                )

    # -----------------------------------
    # Semantic Search
    # -----------------------------------

    st.header("🔍 Semantic Search")

    query = st.text_input(
        "Ask a question about the uploaded documents"
    )

    if st.button("Search"):

        if query.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                db = VectorDatabase()

                if not db.exists():

                    st.error(
                        "❌ Vector Database not found.\n\n"
                        "Please click **Create Vector Database** first."
                    )

                    st.stop()

                db.load()

                results = db.search(
                    query
                )

                st.subheader(
                    "Retrieved Chunks"
                )

                if len(results) == 0:

                    st.warning(
                        "No relevant chunks found."
                    )

                else:

                    for result in results:

                        with st.expander(

                            f"{result.get('chunk_type','Unknown')} - Chunk {result.get('chunk_id','')}"

                        ):

                            st.write(
                                result.get(
                                    "content",
                                    ""
                                )
                            )

                            if "metadata" in result:

                                st.json(
                                    result["metadata"]
                                )

            except Exception as e:

                st.error(
                    f"Search failed.\n\n{e}"
                )