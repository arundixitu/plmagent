from pathlib import Path
import pandas as pd
import json


DATA_FOLDER = Path("data/raw")

OUTPUT_FOLDER = Path("rag/generated")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


documents = []


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def add_document(
    source_file,
    record_id,
    category,
    text,
    metadata=None
):
    """
    Add one semantic document to the engineering RAG corpus.
    """

    if metadata is None:
        metadata = {}

    document = {
        "source_file": source_file,
        "record_id": str(record_id),
        "category": category,
        "text": str(text),
        "metadata": metadata
    }

    documents.append(document)


# ==========================================================
# 1. COMPANY PROFILE
# ==========================================================

file_name = "01_Company_Profile.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for index, row in df.iterrows():

    text_parts = []

    for column in df.columns:

        value = row[column]

        if pd.notna(value):
            text_parts.append(
                f"{column}: {value}"
            )

    add_document(
        source_file=file_name,
        record_id=f"COMPANY-{index + 1}",
        category="Company",
        text=" | ".join(text_parts)
    )


# ==========================================================
# 2. STRATEGY AND PLM OBJECTIVES
# ==========================================================

file_name = "02_Strategy_and_PLM_Objectives.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for index, row in df.iterrows():

    text_parts = []

    for column in df.columns:

        value = row[column]

        if pd.notna(value):
            text_parts.append(
                f"{column}: {value}"
            )

    add_document(
        source_file=file_name,
        record_id=f"STRATEGY-{index + 1}",
        category="Strategy",
        text=" | ".join(text_parts)
    )


# ==========================================================
# 3. PRODUCT FAMILY
# ==========================================================

file_name = "03_Product_Family.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for index, row in df.iterrows():

    if "Product_ID" in df.columns:
        record_id = row["Product_ID"]
    else:
        record_id = f"PRODUCT-{index + 1}"

    text_parts = []

    for column in df.columns:

        value = row[column]

        if pd.notna(value):
            text_parts.append(
                f"{column}: {value}"
            )

    add_document(
        source_file=file_name,
        record_id=record_id,
        category="Product",
        text=" | ".join(text_parts)
    )


# ==========================================================
# 4. ENGINEERING CHANGE HISTORY
# ==========================================================

file_name = "10_Engineering_Changes.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for _, row in df.iterrows():

    text = (
        f"Engineering Change {row['ECR_ID']}. "
        f"Title: {row['Change_Title']}. "
        f"Primary component: {row['Primary_Component_ID']}. "
        f"Change description: {row['Change_Description']}. "
        f"Risk: {row['Risk']}. "
        f"Status: {row['Status']}."
    )

    add_document(
        source_file=file_name,
        record_id=row["ECR_ID"],
        category="Engineering Change",
        text=text,
        metadata={
            "ecr_id": str(row["ECR_ID"]),
            "component_id": str(
                row["Primary_Component_ID"]
            ),
            "risk": str(row["Risk"]),
            "status": str(row["Status"])
        }
    )


# ==========================================================
# 5. HISTORICAL CHANGE IMPACT KNOWLEDGE
# ==========================================================

file_name = "11_Change_Impact_Links.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for _, row in df.iterrows():

    text = (
        f"Historical impact record {row['Impact_ID']} "
        f"for engineering change {row['ECR_ID']}. "
        f"Affected entity: {row['Affected_Entity_ID']}. "
        f"Impact dimension: {row['Impact_Dimension']}. "
        f"Propagation level: {row['Propagation_Level']}. "
        f"Status: {row['Status']}. "
        f"Rationale: {row['Rationale']}. "
        f"Recommended action: "
        f"{row['Recommended_Action']}."
    )

    add_document(
        source_file=file_name,
        record_id=row["Impact_ID"],
        category="Historical Impact",
        text=text,
        metadata={
            "ecr_id": str(row["ECR_ID"]),
            "affected_entity": str(
                row["Affected_Entity_ID"]
            ),
            "impact_dimension": str(
                row["Impact_Dimension"]
            ),
            "propagation_level": str(
                row["Propagation_Level"]
            )
        }
    )


# ==========================================================
# 6. SERVICE KNOWLEDGE
# ==========================================================

file_name = "14_Service_Records.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for _, row in df.iterrows():

    text = (
        f"Service record {row['Service_ID']}. "
        f"Component: {row['Component_ID']}. "
        f"Issue: {row['Issue']}. "
        f"Service action: {row['Service_Action']}. "
        f"Status: {row['Status']}. "
        f"Related engineering change: "
        f"{row['Related_ECR']}."
    )

    add_document(
        source_file=file_name,
        record_id=row["Service_ID"],
        category="Service",
        text=text,
        metadata={
            "component_id": str(
                row["Component_ID"]
            ),
            "related_ecr": str(
                row["Related_ECR"]
            )
        }
    )


# ==========================================================
# 7. FIELD FAILURE KNOWLEDGE
# ==========================================================

file_name = "15_Field_Failure_Records.xlsx"

df = pd.read_excel(
    DATA_FOLDER / file_name
)

for _, row in df.iterrows():

    text = (
        f"Field failure record {row['Field_ID']}. "
        f"Component: {row['Component_ID']}. "
        f"Failure mode: {row['Failure_Mode']}. "
        f"Operating context: {row['Context']}. "
        f"Severity: {row['Severity']}. "
        f"Related engineering change: "
        f"{row['Related_ECR']}."
    )

    add_document(
        source_file=file_name,
        record_id=row["Field_ID"],
        category="Field Failure",
        text=text,
        metadata={
            "component_id": str(
                row["Component_ID"]
            ),
            "severity": str(
                row["Severity"]
            ),
            "related_ecr": str(
                row["Related_ECR"]
            )
        }
    )


# ==========================================================
# SAVE ENGINEERING RAG CORPUS
# ==========================================================

output_file = (
    OUTPUT_FOLDER /
    "engineering_rag_documents_v1.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        documents,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================================
# REPORT
# ==========================================================

print("\nAUREON ENGINEERING RAG DOCUMENT BUILDER")
print("=" * 90)

print(
    "Documents generated:",
    len(documents)
)


print("\nDOCUMENTS BY CATEGORY")
print("-" * 90)

category_counts = {}

for document in documents:

    category = document["category"]

    category_counts[category] = (
        category_counts.get(category, 0) + 1
    )


for category, count in category_counts.items():

    print(
        f"{category}: {count}"
    )


# ==========================================================
# FILE POLICY REPORT
# ==========================================================

print("\nCORPUS POLICY")
print("-" * 90)

print(
    "Semantic RAG: company, strategy, product, "
    "engineering-change history, historical impacts, "
    "service and field failures."
)

print(
    "Structured tools: components, requirements, BOM, "
    "specifications, suppliers, manufacturing, validation, "
    "document control and relationship matrices."
)

print(
    "Learning corpus: Student Learning Cases will be "
    "handled separately."
)


# ==========================================================
# SECURITY CHECK
# ==========================================================

print("\nSECURITY CHECK")
print("-" * 90)

forbidden_files = [
    "99_Ground_Truth_Impact_Matrix.xlsx",
    "Aureon_EMove_X1_PLM_Agent_Dataset_V1_COMPLETE.xlsx"
]

security_pass = True


for forbidden_file in forbidden_files:

    used = any(
        document["source_file"] == forbidden_file
        for document in documents
    )

    if used:

        print(
            "ERROR - Forbidden file indexed:",
            forbidden_file
        )

        security_pass = False

    else:

        print(
            "PASS - Excluded from RAG:",
            forbidden_file
        )


# ==========================================================
# STRUCTURED-DATA EXCLUSION CHECK
# ==========================================================

structured_only_files = [
    "04_Component_Master.xlsx",
    "05_Product_Relationships.xlsx",
    "06_Product_Requirements.xlsx",
    "07_BOM.xlsx",
    "08_Design_Specifications.xlsx",
    "09_Supplier_Records.xlsx",
    "12_Manufacturing_Records.xlsx",
    "13_Test_Validation_Records.xlsx",
    "16_Document_Control.xlsx",
    "17_Data_Dictionary.xlsx",
    "18_Relationship_Matrix.xlsx",
    "19_Agent_Tool_Registry.xlsx"
]


print("\nSTRUCTURED TOOL DATA CHECK")
print("-" * 90)


for structured_file in structured_only_files:

    used = any(
        document["source_file"] == structured_file
        for document in documents
    )

    if used:

        print(
            "WARNING - Structured file entered RAG:",
            structured_file
        )

    else:

        print(
            "PASS - Reserved for structured/tool layer:",
            structured_file
        )


# ==========================================================
# FINAL STATUS
# ==========================================================

print("\nOUTPUT")
print("-" * 90)

print(output_file)


print("\nFINAL STATUS")
print("=" * 90)

if security_pass:

    print(
        "PASS - Engineering RAG corpus created "
        "without evaluator leakage."
    )

else:

    print(
        "FAIL - Forbidden evaluator information "
        "detected in RAG corpus."
    )