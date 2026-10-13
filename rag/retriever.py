from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import chromadb


VECTOR_FOLDER = Path("rag/vector_store")
COLLECTION_NAME = "aureon_engineering_rag_v1"


class AureonRetriever:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_FOLDER)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def retrieve_evidence(
        self,
        query,
        top_k=8,
        exclude_ecr=None
    ):
        """
        Retrieve semantic PLM evidence.

        exclude_ecr:
            Used during benchmarking to prevent
            retrieval of evidence belonging directly
            to the ECR under evaluation.
        """

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        # Retrieve more than needed because some records
        # may later be excluded.
        search_size = min(
            self.collection.count(),
            top_k * 4
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=search_size,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            # ------------------------------------------
            # BENCHMARK LEAKAGE CONTROL
            # ------------------------------------------

            if exclude_ecr is not None:

                metadata_ecr = metadata.get("ecr_id")

                related_ecr = metadata.get(
                    "related_ecr"
                )

                record_id = metadata.get(
                    "record_id"
                )

                # Historical impact belonging to test ECR
                if metadata_ecr == exclude_ecr:
                    continue

                # Service / field record explicitly linked
                # to test ECR
                if related_ecr == exclude_ecr:
                    continue

                # The ECR record itself
                if record_id == exclude_ecr:
                    continue

            retrieved.append(
                {
                    "record_id": metadata.get(
                        "record_id"
                    ),
                    "category": metadata.get(
                        "category"
                    ),
                    "source_file": metadata.get(
                        "source_file"
                    ),
                    "distance": float(distance),
                    "text": document,
                    "metadata": metadata
                }
            )

            if len(retrieved) >= top_k:
                break

        return retrieved