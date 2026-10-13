import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DOCUMENT_FILE = Path(
    "rag/generated/engineering_rag_documents_v1.json"
)

VECTOR_FOLDER = Path(
    "rag/vector_store"
)

COLLECTION_NAME = "aureon_engineering_rag_v1"


print("\nAUREON VECTOR STORE BUILDER")
print("=" * 90)


# ----------------------------------------------------------
# LOAD DOCUMENTS
# ----------------------------------------------------------

with open(
    DOCUMENT_FILE,
    "r",
    encoding="utf-8"
) as file:

    documents = json.load(file)


print("Documents loaded:", len(documents))


# ----------------------------------------------------------
# LOAD EMBEDDING MODEL
# ----------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ----------------------------------------------------------
# CREATE CHROMA DATABASE
# ----------------------------------------------------------

VECTOR_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=str(VECTOR_FOLDER)
)


# Remove old collection if it exists
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME
)


# ----------------------------------------------------------
# PREPARE DOCUMENTS
# ----------------------------------------------------------

ids = []
texts = []
metadatas = []


for index, document in enumerate(documents):

    document_id = (
        f"{document['category']}_"
        f"{document['record_id']}_"
        f"{index}"
    )

    metadata = {
        "source_file": document["source_file"],
        "record_id": document["record_id"],
        "category": document["category"]
    }

    # Add additional metadata
    for key, value in document.get(
        "metadata",
        {}
    ).items():

        metadata[key] = str(value)

    ids.append(document_id)
    texts.append(document["text"])
    metadatas.append(metadata)


# ----------------------------------------------------------
# CREATE EMBEDDINGS
# ----------------------------------------------------------

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

embeddings = embeddings.tolist()


print(
    "Embeddings created:",
    len(embeddings)
)


# ----------------------------------------------------------
# STORE IN CHROMA
# ----------------------------------------------------------

print("\nWriting vectors to Chroma...")

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings
)


print(
    "Vectors stored:",
    collection.count()
)


# ----------------------------------------------------------
# SECURITY CHECK
# ----------------------------------------------------------

print("\nSECURITY CHECK")
print("-" * 90)

forbidden_sources = {
    "99_Ground_Truth_Impact_Matrix.xlsx",
    "Aureon_EMove_X1_PLM_Agent_Dataset_V1_COMPLETE.xlsx"
}


security_pass = True


for metadata in metadatas:

    source = metadata["source_file"]

    if source in forbidden_sources:

        print(
            "ERROR - Forbidden source detected:",
            source
        )

        security_pass = False


if security_pass:

    print(
        "PASS - No evaluator-only source "
        "stored in vector database."
    )


# ----------------------------------------------------------
# FINAL REPORT
# ----------------------------------------------------------

print("\nVECTOR STORE")
print("-" * 90)

print("Collection:", COLLECTION_NAME)
print("Location:", VECTOR_FOLDER)
print("Vector count:", collection.count())


print("\nFINAL STATUS")
print("=" * 90)

if security_pass:

    print(
        "PASS - Aureon engineering vector store "
        "created successfully."
    )

else:

    print(
        "FAIL - Security policy violation detected."
    )