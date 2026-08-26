"""End-to-end check that retrieval is really restricted to the selected document.

Run against a PostgreSQL instance (no Gemini API key required):

    POSTGRES_HOST=localhost POSTGRES_DB=enterprise_chunking \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres \
    python test_selected_document_retrieval.py

It seeds two documents, then verifies that:

1. A generic question ("what this document about?") about the selected
   document returns chunks (> 0) before any answer generation.
2. Every returned chunk belongs to the selected document only.
3. Selecting the other document never returns the first document's chunks.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")


def load_app_module():
    """Import app.py without running its Streamlit page routing."""

    source = Path(__file__).with_name("app.py").read_text()

    marker = "# MAIN APPLICATION"
    head = source.split(marker)[0]

    module = types.ModuleType("app_under_test")
    module.__file__ = str(Path(__file__).with_name("app.py"))

    sys.modules["app_under_test"] = module

    exec(compile(head, module.__file__, "exec"), module.__dict__)

    return module


app = load_app_module()


DOCUMENT_A = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
DOCUMENT_B = "Quarterly-Revenue-Report.pdf"

DOCUMENT_A_CHUNKS = [
    (
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks that include an encoder "
        "and a decoder. We propose a new simple network architecture, the "
        "Transformer, based solely on attention mechanisms."
    ),
    (
        "Multi-head attention allows the model to jointly attend to "
        "information from different representation subspaces at different "
        "positions."
    ),
]

DOCUMENT_B_CHUNKS = [
    (
        "Revenue for the third quarter increased by twelve percent "
        "year over year, driven by subscription growth."
    ),
]


def seed(db, user_id, filename, contents):
    conn = db.connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO public.documents
        (user_id, filename, file_type, file_size, chunking_method)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (user_id, filename, "pdf", 1024, "Recursive"),
    )

    document_id = int(cursor.fetchone()[0])

    for index, content in enumerate(contents):
        cursor.execute(
            """
            INSERT INTO public.document_chunks
            (document_id, chunk_id, chunk_type, content, page,
             tokens, characters)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                index,
                "Recursive",
                content,
                str(index + 1),
                len(content.split()),
                len(content),
            ),
        )

    conn.commit()
    cursor.close()

    return document_id


def cleanup(db, user_id):
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM public.users WHERE id = %s",
        (user_id,),
    )
    conn.commit()
    cursor.close()


class RecordingLLM:
    """Planner that always wants the web, so the scope must override it."""

    def __init__(self):
        self.answer_chunks = None
        self.web_search_called = False

    def plan_action(self, query, previous_actions=None,
                    previous_evaluations=None):
        return {
            "action": "web_search",
            "query": query,
            "reason": "test planner prefers the web",
        }

    def evaluate_evidence(self, query, results, action=None):
        return {"sufficient": False, "confidence": 0.0}

    def web_search(self, query, max_results=5):
        self.web_search_called = True
        return [{"content": "web noise", "url": "https://example.com"}]

    def generate_answer(self, query, chunks, source_type="document"):
        self.answer_chunks = chunks
        return "grounded answer"

    def generate(self, query):
        raise AssertionError(
            "Direct model answering must not happen for a selected document."
        )


def check_pipeline(db, user_id, document_a, document_b):
    """Run perform_rag() with a document selected."""

    failures = []

    st = app.st

    llm = RecordingLLM()

    st.session_state.db = db
    st.session_state.user = {"id": user_id, "username": "retrieval_test_user"}
    st.session_state.llm = llm
    st.session_state.top_k = 5
    st.session_state.selected_chunk_types = []
    st.session_state.selected_document_ids = [document_a]

    answer, sources, source_type = app.perform_rag(
        "what this document about?"
    )

    print("PIPELINE SOURCE TYPE:", source_type)
    print("PIPELINE CHUNKS:", len(sources))

    if source_type != "documents" or not sources:
        failures.append(
            "perform_rag did not answer from the selected document "
            f"(source_type={source_type}, chunks={len(sources)})."
        )

    if llm.web_search_called:
        failures.append(
            "perform_rag ran a web search even though a document was "
            "selected."
        )

    if any(c.get("document_id") != document_a for c in sources):
        failures.append(
            "perform_rag returned evidence from a non-selected document."
        )

    if not llm.answer_chunks:
        failures.append(
            "Gemini would have been called without any retrieved chunks."
        )

    # A document with no indexed chunks must produce an explicit
    # application-level message, not "you forgot to attach the document".

    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO public.documents
        (user_id, filename, file_type, file_size, chunking_method)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (user_id, "Unindexed.pdf", "pdf", 10, "Recursive"),
    )
    empty_document = int(cursor.fetchone()[0])
    conn.commit()
    cursor.close()

    st.session_state.selected_document_ids = [empty_document]

    answer, sources, source_type = app.perform_rag(
        "what this document about?"
    )

    print("UNINDEXED ANSWER:", answer)

    if source_type != "error" or "re-index" not in answer.lower():
        failures.append(
            "A selected but unindexed document did not produce an "
            f"indexing error message (got: {answer!r})."
        )

    # Switching the selection must not reuse the previous document.

    st.session_state.selected_document_ids = [document_b]

    answer, sources, source_type = app.perform_rag(
        "what this document about?"
    )

    if any(c.get("document_id") == document_a for c in sources):
        failures.append(
            "perform_rag leaked the previously selected document."
        )

    return failures


def main():
    db = app.PostgreSQLStore()
    db.initialize()

    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO public.users (username, password_hash, role)
        VALUES (%s, %s, 'user')
        RETURNING id
        """,
        ("retrieval_test_user", "x"),
    )
    user_id = int(cursor.fetchone()[0])
    conn.commit()
    cursor.close()

    failures = []

    try:
        document_a = seed(db, user_id, DOCUMENT_A, DOCUMENT_A_CHUNKS)
        document_b = seed(db, user_id, DOCUMENT_B, DOCUMENT_B_CHUNKS)

        question = "what this document about?"

        # ------------------------------------------------------------
        # 1. Selected document must produce evidence.
        # ------------------------------------------------------------

        chunks = db.search_chunks(
            user_id=user_id,
            query=question,
            selected_document_ids=[document_a],
            top_k=5,
        )

        print("SELECTED DOCUMENT:", DOCUMENT_A)
        print("USER QUERY:", question)
        print("RETRIEVED CHUNKS:", len(chunks))
        print(
            "RETRIEVED FILENAMES:",
            sorted({c.get("filename") for c in chunks}),
        )

        if not chunks:
            failures.append(
                "Selected document returned zero chunks for a generic "
                "question."
            )

        if any(c.get("document_id") != document_a for c in chunks):
            failures.append(
                "Retrieval leaked chunks from a non-selected document."
            )

        # ------------------------------------------------------------
        # 2. Switching documents must not return the previous document.
        # ------------------------------------------------------------

        other = db.search_chunks(
            user_id=user_id,
            query=question,
            selected_document_ids=[document_b],
            top_k=5,
        )

        print("SELECTED DOCUMENT:", DOCUMENT_B)
        print("RETRIEVED CHUNKS:", len(other))
        print(
            "RETRIEVED FILENAMES:",
            sorted({c.get("filename") for c in other}),
        )

        if not other:
            failures.append(
                "Second document returned zero chunks when selected."
            )

        if any(c.get("document_id") == document_a for c in other):
            failures.append(
                "Chunks from the previously selected document leaked into "
                "the new selection."
            )

        # ------------------------------------------------------------
        # 3. A keyword question still works and stays filtered.
        # ------------------------------------------------------------

        keyword = db.search_chunks(
            user_id=user_id,
            query="multi-head attention",
            selected_document_ids=[document_a],
            top_k=5,
        )

        print("KEYWORD QUERY CHUNKS:", len(keyword))

        if not keyword:
            failures.append(
                "Keyword search inside the selected document returned "
                "nothing."
            )

        if any(c.get("document_id") != document_a for c in keyword):
            failures.append(
                "Keyword search leaked chunks from another document."
            )

        # ------------------------------------------------------------
        # 4. Full pipeline: the selection must reach the answer step.
        # ------------------------------------------------------------

        failures.extend(
            check_pipeline(db, user_id, document_a, document_b)
        )

        # ------------------------------------------------------------
        # 5. No selection still searches everything.
        # ------------------------------------------------------------

        unfiltered = db.search_chunks(
            user_id=user_id,
            query="revenue",
            selected_document_ids=None,
            top_k=5,
        )

        print("UNFILTERED CHUNKS:", len(unfiltered))

        if not any(
            c.get("document_id") == document_b for c in unfiltered
        ):
            failures.append(
                "Unfiltered search did not reach all documents."
            )

    finally:
        cleanup(db, user_id)

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(" -", failure)
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
