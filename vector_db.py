import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, VECTOR_DB_PATH


class VectorDatabase:

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

    # --------------------------------------------------
    # Generate Embeddings
    # --------------------------------------------------

    def generate_embeddings(self, chunks):

        if len(chunks) == 0:

            raise ValueError(
                "No chunks available for embedding."
            )

        texts = [
            chunk.get(
                "content",
                ""
            )
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.astype("float32")

    # --------------------------------------------------
    # Create Index
    # --------------------------------------------------

    def create_index(self, chunks):

        embeddings = self.generate_embeddings(
            chunks
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

        self.documents = chunks

        print(
            f"✅ Indexed {len(chunks)} chunks."
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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
        ) as f:

            pickle.dump(
                self.documents,
                f
            )

        print(
            "✅ Vector Database Saved."
        )

    # --------------------------------------------------
    # Exists
    # --------------------------------------------------

    def exists(self):

        return (
            os.path.isfile(self.index_path)
            and
            os.path.isfile(self.documents_path)
        )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load(self):

        if not self.exists():

            raise FileNotFoundError(
                "Vector Database not found. Please create it first."
            )

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.documents_path,
            "rb"
        ) as f:

            self.documents = pickle.load(
                f
            )

        print(
            "✅ Vector Database Loaded."
        )

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=5
    ):

        if self.index is None:

            self.load()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx == -1:

                continue

            if idx < len(self.documents):

                results.append(
                    self.documents[idx]
                )

        return results

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    def display_results(
        self,
        results
    ):

        print("=" * 60)

        for chunk in results:

            print(
                f"\nChunk ID : {chunk.get('chunk_id')}"
            )

            print(
                f"Type     : {chunk.get('chunk_type')}"
            )

            print(
                chunk.get(
                    "content",
                    ""
                )[:500]
            )

            print("-" * 60)