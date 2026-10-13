from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_FOLDER = Path("rag/vector_store")

COLLECTION_NAME = "aureon_engineering_rag_v1"


print("\nAUREON RAG RETRIEVAL TEST")
print("=" * 90)


# ----------------------------------------------------------
# LOAD EMBEDDING MODEL
# ----------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ----------------------------------------------------------
# CONNECT TO CHROMA
# ----------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_FOLDER)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


print(
    "Documents available:",
    collection.count()
)


# ----------------------------------------------------------
# TEST QUERY
# ----------------------------------------------------------

query = (
    "What previous problems were associated "
    "with battery range and battery performance?"
)


print("\nQUERY")
print("-" * 90)

print(query)


# ----------------------------------------------------------
# CREATE QUERY EMBEDDING
# ----------------------------------------------------------

query_embedding = embedding_model.encode(
    [query]
).tolist()


# ----------------------------------------------------------
# SEARCH
# ----------------------------------------------------------

results = collection.query(
    query_embeddings=query_embedding,
    n_results=8,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# ----------------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------------

print("\nTOP RETRIEVED EVIDENCE")
print("=" * 90)


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


for index, (
    document,
    metadata,
    distance
) in enumerate(
    zip(
        documents,
        metadatas,
        distances
    ),
    start=1
):

    print(f"\nRESULT {index}")
    print("-" * 90)

    print(
        "Category:",
        metadata.get("category")
    )

    print(
        "Record ID:",
        metadata.get("record_id")
    )

    print(
        "Source:",
        metadata.get("source_file")
    )

    print(
        "Distance:",
        round(distance, 4)
    )

    print(
        "Evidence:",
        document
    )


print("\nRETRIEVAL TEST COMPLETE.")