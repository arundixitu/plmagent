import pandas as pd


impact_file = "data/raw/11_Change_Impact_Links.xlsx"
ground_truth_file = "data/raw/99_Ground_Truth_Impact_Matrix.xlsx"


impacts = pd.read_excel(impact_file)
ground_truth = pd.read_excel(ground_truth_file)


print("\nAUREON IMPACT vs GROUND TRUTH AUDIT")
print("=" * 100)


total_missing = 0
total_extra = 0
exact_matches = 0


for _, gt_row in ground_truth.iterrows():

    ecr_id = gt_row["ECR_ID"]

    # --------------------------------------------------
    # Ground-truth entities
    # --------------------------------------------------

    expected_text = str(
        gt_row["Expected_Impacted_Entities"]
    )

    expected_entities = {
        item.strip()
        for item in expected_text.split(";")
        if item.strip()
    }


    # --------------------------------------------------
    # Production impact entities
    # --------------------------------------------------

    production_rows = impacts[
        impacts["ECR_ID"] == ecr_id
    ]

    production_entities = set(
        production_rows[
            "Affected_Entity_ID"
        ].dropna().astype(str)
    )


    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    missing = (
        expected_entities
        - production_entities
    )

    extra = (
        production_entities
        - expected_entities
    )

    matched = (
        expected_entities
        & production_entities
    )


    print("\n")
    print("=" * 100)
    print(ecr_id)
    print("=" * 100)

    print(
        "Ground truth count:",
        len(expected_entities)
    )

    print(
        "Production impact count:",
        len(production_entities)
    )

    print(
        "Matched:",
        len(matched)
    )

    print(
        "Missing from production:",
        len(missing)
    )

    print(
        "Extra in production:",
        len(extra)
    )


    if missing:

        print("\nMISSING")
        print("-" * 100)

        for entity in sorted(missing):
            print(entity)


    if extra:

        print("\nEXTRA")
        print("-" * 100)

        for entity in sorted(extra):
            print(entity)


    if not missing and not extra:

        print("\nEXACT MATCH")

        exact_matches += 1


    total_missing += len(missing)
    total_extra += len(extra)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n")
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    "Ground-truth ECRs:",
    len(ground_truth)
)

print(
    "Exact matches:",
    exact_matches
)

print(
    "Total missing entities:",
    total_missing
)

print(
    "Total extra entities:",
    total_extra
)


if total_missing == 0 and total_extra == 0:

    print(
        "\nPASS - Production impact links exactly match "
        "the evaluator ground truth."
    )

else:

    print(
        "\nREVIEW REQUIRED - Production impact links and "
        "ground truth are not identical."
    )