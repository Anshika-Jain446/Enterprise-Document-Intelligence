import os
import pickle

import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from config import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)


class VectorDatabase:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.index = None

        self.documents = []

        os.makedirs(
            VECTOR_DB_PATH,
            exist_ok=True
        )

        self.index_path = os.path.join(
            VECTOR_DB_PATH,
            "faiss.index"
        )

        self.documents_path = os.path.join(
            VECTOR_DB_PATH,
            "documents.pkl"
        )

    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    def generate_embeddings(
        self,
        chunks
    ):

        if not chunks:

            raise ValueError(
                "No chunks available "
                "for embedding."
            )

        texts = [
            str(
                chunk.get(
                    "content",
                    ""
                )
            )
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.astype(
            "float32"
        )

    # ========================================================
    # CREATE INDEX
    # ========================================================

    def create_index(
        self,
        chunks
    ):

        if not chunks:

            raise ValueError(
                "No chunks available "
                "to index."
            )

        embeddings = (
            self.generate_embeddings(
                chunks
            )
        )

        dimension = (
            embeddings.shape[1]
        )

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

        self.documents = list(
            chunks
        )

        print(
            f"Indexed {len(chunks)} chunks."
        )

    # ========================================================
    # SAVE DATABASE
    # ========================================================

    def save(self):

        if self.index is None:

            raise RuntimeError(
                "No FAISS index to save."
            )

        os.makedirs(
            VECTOR_DB_PATH,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.documents_path,
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )

        print(
            "Vector Database Saved."
        )

    # ========================================================
    # DATABASE EXISTS
    # ========================================================

    def exists(self):

        return (
            os.path.isfile(
                self.index_path
            )
            and
            os.path.isfile(
                self.documents_path
            )
        )

    # ========================================================
    # LOAD DATABASE
    # ========================================================

    def load(self):

        if not self.exists():

            raise FileNotFoundError(
                "Vector Database not found. "
                "Please create it first."
            )

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.documents_path,
            "rb"
        ) as file:

            self.documents = (
                pickle.load(file)
            )

        if not isinstance(
            self.documents,
            list
        ):

            self.documents = list(
                self.documents
            )

        print(
            "Vector Database Loaded."
        )

    # ========================================================
    # DATABASE STATISTICS
    # ========================================================

    def get_stats(self):

        stats = {
            "status":
                "Not Created",

            "chunks":
                0,

            "dimension":
                0,

            "documents":
                0,

            "index_type":
                "FAISS IndexFlatL2",

            "path":
                VECTOR_DB_PATH,
        }

        if not self.exists():

            return stats

        try:

            index = faiss.read_index(
                self.index_path
            )

            with open(
                self.documents_path,
                "rb"
            ) as file:

                documents = (
                    pickle.load(file)
                )

            stats[
                "status"
            ] = "Ready"

            stats[
                "chunks"
            ] = int(
                index.ntotal
            )

            stats[
                "dimension"
            ] = int(
                index.d
            )

            stats[
                "documents"
            ] = len(
                documents
            )

        except Exception as e:

            stats[
                "status"
            ] = "Error"

            stats[
                "error"
            ] = str(e)

        return stats

    # ========================================================
    # CLEAR DATABASE
    # ========================================================

    def clear(self):

        if os.path.exists(
            self.index_path
        ):

            os.remove(
                self.index_path
            )

        if os.path.exists(
            self.documents_path
        ):

            os.remove(
                self.documents_path
            )

        self.index = None

        self.documents = []

        print(
            "Vector Database Cleared."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5
    ):

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if self.index is None:

            self.load()

        if self.index is None:

            return []

        if self.index.ntotal == 0:

            return []

        top_k = max(
            1,
            min(
                int(top_k),
                self.index.ntotal
            )
        )

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True
            )
            .astype("float32")
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index < 0:

                continue

            if index >= len(
                self.documents
            ):

                continue

            chunk = dict(
                self.documents[index]
            )

            distance = float(
                distance
            )

            similarity_score = (
                1.0
                /
                (
                    1.0
                    +
                    max(
                        distance,
                        0.0
                    )
                )
            )

            chunk[
                "similarity_score"
            ] = similarity_score

            chunk[
                "distance"
            ] = distance

            results.append(
                chunk
            )

        return results

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
        results
    ):

        print(
            "=" * 60
        )

        if not results:

            print(
                "No results found."
            )

            print(
                "=" * 60
            )

            return

        for chunk in results:

            print(
                f"\nChunk ID : "
                f"{chunk.get('chunk_id', 'N/A')}"
            )

            print(
                f"Type     : "
                f"{chunk.get('chunk_type', 'N/A')}"
            )

            if (
                "similarity_score"
                in chunk
            ):

                print(
                    f"Score    : "
                    f"{chunk['similarity_score']:.4f}"
                )

            if "distance" in chunk:

                print(
                    f"Distance : "
                    f"{chunk['distance']:.4f}"
                )

            print(
                str(
                    chunk.get(
                        "content",
                        ""
                    )
                )[:500]
            )

            print(
                "-" * 60
            )