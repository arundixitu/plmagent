from retriever import AureonRetriever


retriever = AureonRetriever()


query = (
    "What historical engineering evidence is relevant "
    "to increasing battery capacity?"
)


print("\nAUREON BENCHMARK-SAFE RETRIEVAL TEST")
print("=" * 90)


print("\nQUERY")
print("-" * 90)

print(query)


print("\nTEST CASE")
print("-" * 90)

print(
    "Assume ECR-001 is currently being evaluated."
)

print(
    "Evidence explicitly belonging to ECR-001 "
    "must therefore be hidden."
)


results = retriever.retrieve_evidence(
    query=query,
    top_k=10,
    exclude_ecr="ECR-001"
)


print("\nRETRIEVED EVIDENCE")
print("=" * 90)


for number, result in enumerate(
    results,
    start=1
):

    print(f"\nRESULT {number}")
    print("-" * 90)

    print(
        "Record:",
        result["record_id"]
    )

    print(
        "Category:",
        result["category"]
    )

    print(
        "Distance:",
        round(
            result["distance"],
            4
        )
    )

    print(
        "Evidence:",
        result["text"]
    )


print("\nLEAKAGE CHECK")
print("=" * 90)


leakage_found = False


for result in results:

    metadata = result["metadata"]

    if metadata.get("ecr_id") == "ECR-001":
        leakage_found = True

    if metadata.get("related_ecr") == "ECR-001":
        leakage_found = True

    if result["record_id"] == "ECR-001":
        leakage_found = True


if leakage_found:

    print(
        "FAIL - ECR-001 evidence leaked into retrieval."
    )

else:

    print(
        "PASS - Direct ECR-001 evidence successfully excluded."
    )