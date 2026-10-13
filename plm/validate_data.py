from data_engine import PLMDataEngine


plm = PLMDataEngine()

errors = []
warnings = []


# ==========================================================
# CONFIGURATION
# ==========================================================

# These ECRs intentionally have no production impact records.
# They are reserved as open discovery/demo cases.
discovery_ecrs = {
    "ECR-002",
    "ECR-003",
    "ECR-010",
    "ECR-011",
    "ECR-014",
}


# ==========================================================
# 1. COMPONENT IDS
# ==========================================================

component_ids = set(
    plm.components["Component_ID"]
    .dropna()
    .astype(str)
)


# ==========================================================
# 2. PRODUCT IDS
# ==========================================================

product_ids = {
    "PROD-X1",
    "PROD-X2",
    "PROD-X1-LR",
    "PROD-X1-C",
}

valid_product_entities = component_ids.union(product_ids)


# ==========================================================
# 3. PRODUCT RELATIONSHIP CHECK
# ==========================================================

for _, row in plm.relationships.iterrows():

    relationship_id = row["Relationship_ID"]
    source = str(row["Source_ID"])
    target = str(row["Target_ID"])

    if source not in valid_product_entities:
        errors.append(
            f"{relationship_id} has invalid source entity {source}"
        )

    if target not in valid_product_entities:
        errors.append(
            f"{relationship_id} has invalid target entity {target}"
        )


# ==========================================================
# 4. ENGINEERING CHANGE PRIMARY COMPONENT CHECK
# ==========================================================

ecr_ids = set(
    plm.changes["ECR_ID"]
    .dropna()
    .astype(str)
)

for _, row in plm.changes.iterrows():

    ecr_id = str(row["ECR_ID"])
    component_id = str(row["Primary_Component_ID"])

    if component_id not in component_ids:
        errors.append(
            f"{ecr_id} references missing primary component "
            f"{component_id}"
        )


# ==========================================================
# 5. REQUIREMENT IDS
# ==========================================================

requirement_ids = set(
    plm.requirements["Requirement_ID"]
    .dropna()
    .astype(str)
)


# ==========================================================
# 6. VALIDATION -> REQUIREMENT CHECK
# ==========================================================

validation_ids = set(
    plm.validation["Test_ID"]
    .dropna()
    .astype(str)
)

for _, row in plm.validation.iterrows():

    test_id = str(row["Test_ID"])
    requirement_id = str(row["Requirement_ID"])

    if requirement_id not in requirement_ids:
        errors.append(
            f"{test_id} references missing requirement "
            f"{requirement_id}"
        )


# ==========================================================
# 7. MANUFACTURING -> COMPONENT CHECK
# ==========================================================

manufacturing_ids = set(
    plm.manufacturing["Manufacturing_ID"]
    .dropna()
    .astype(str)
)

for _, row in plm.manufacturing.iterrows():

    manufacturing_id = str(row["Manufacturing_ID"])
    component_id = str(row["Component_ID"])

    if component_id not in component_ids:
        errors.append(
            f"{manufacturing_id} references missing component "
            f"{component_id}"
        )


# ==========================================================
# 8. SERVICE -> COMPONENT / ECR CHECK
# ==========================================================

service_ids = set(
    plm.service["Service_ID"]
    .dropna()
    .astype(str)
)

for _, row in plm.service.iterrows():

    service_id = str(row["Service_ID"])
    component_id = str(row["Component_ID"])
    related_ecr = row["Related_ECR"]

    if component_id not in component_ids:
        errors.append(
            f"{service_id} references missing component "
            f"{component_id}"
        )

    if related_ecr is not None:

        related_ecr_text = str(related_ecr)

        if related_ecr_text.lower() != "nan":

            if related_ecr_text not in ecr_ids:
                errors.append(
                    f"{service_id} references missing ECR "
                    f"{related_ecr_text}"
                )


# ==========================================================
# 9. SUPPLIER -> COMPONENT CHECK
# ==========================================================

supplier_ids = set(
    plm.suppliers["Supplier_ID"]
    .dropna()
    .astype(str)
)

for _, row in plm.suppliers.iterrows():

    supplier_id = str(row["Supplier_ID"])
    component_id = str(row["Primary_Component_ID"])

    if component_id not in component_ids:
        errors.append(
            f"{supplier_id} references missing component "
            f"{component_id}"
        )


# ==========================================================
# 10. VALID IMPACT ENTITY SET
# ==========================================================

valid_impact_entities = (
    component_ids
    .union(requirement_ids)
    .union(validation_ids)
    .union(manufacturing_ids)
    .union(service_ids)
    .union(supplier_ids)
)


# ==========================================================
# 11. CHANGE IMPACT LINK CHECK
# ==========================================================

for _, row in plm.change_impacts.iterrows():

    impact_id = str(row["Impact_ID"])
    ecr_id = str(row["ECR_ID"])
    entity_id = str(row["Affected_Entity_ID"])

    if ecr_id not in ecr_ids:
        errors.append(
            f"{impact_id} references missing ECR {ecr_id}"
        )

    if entity_id not in valid_impact_entities:
        errors.append(
            f"{impact_id} references missing entity {entity_id}"
        )


# ==========================================================
# 12. DUPLICATE ID CHECKS
# ==========================================================

datasets = {
    "Component_ID": plm.components["Component_ID"],
    "Requirement_ID": plm.requirements["Requirement_ID"],
    "ECR_ID": plm.changes["ECR_ID"],
    "Impact_ID": plm.change_impacts["Impact_ID"],
    "Manufacturing_ID": plm.manufacturing["Manufacturing_ID"],
    "Test_ID": plm.validation["Test_ID"],
    "Service_ID": plm.service["Service_ID"],
    "Supplier_ID": plm.suppliers["Supplier_ID"],
}

for id_name, series in datasets.items():

    cleaned = series.dropna().astype(str)

    duplicates = cleaned[
        cleaned.duplicated()
    ].tolist()

    if duplicates:
        errors.append(
            f"Duplicate {id_name}: {duplicates}"
        )


# ==========================================================
# 13. REQUIRED COLUMN CHECKS
# ==========================================================

required_columns = {

    "Components": (
        plm.components,
        [
            "Component_ID",
            "Component_Name",
            "Category",
        ],
    ),

    "Requirements": (
        plm.requirements,
        [
            "Requirement_ID",
            "Product_ID",
            "Requirement_Text",
        ],
    ),

    "Relationships": (
        plm.relationships,
        [
            "Relationship_ID",
            "Source_ID",
            "Target_ID",
            "Relationship_Type",
        ],
    ),

    "Engineering Changes": (
        plm.changes,
        [
            "ECR_ID",
            "Primary_Component_ID",
            "Change_Title",
            "Change_Description",
        ],
    ),

    "Change Impacts": (
        plm.change_impacts,
        [
            "Impact_ID",
            "ECR_ID",
            "Affected_Entity_ID",
            "Impact_Dimension",
            "Propagation_Level",
        ],
    ),

    "Manufacturing": (
        plm.manufacturing,
        [
            "Manufacturing_ID",
            "Component_ID",
            "Process_Name",
        ],
    ),

    "Validation": (
        plm.validation,
        [
            "Test_ID",
            "Requirement_ID",
            "Test_Name",
        ],
    ),

    "Service": (
        plm.service,
        [
            "Service_ID",
            "Component_ID",
            "Issue",
        ],
    ),

    "Suppliers": (
        plm.suppliers,
        [
            "Supplier_ID",
            "Supplier_Name",
            "Primary_Component_ID",
        ],
    ),
}


for dataset_name, dataset_info in required_columns.items():

    dataframe = dataset_info[0]
    columns = dataset_info[1]

    for column in columns:

        if column not in dataframe.columns:
            errors.append(
                f"{dataset_name} missing required column {column}"
            )


# ==========================================================
# 14. MISSING CRITICAL VALUE CHECK
# ==========================================================

critical_columns = {

    "Components": (
        plm.components,
        [
            "Component_ID",
            "Component_Name",
        ],
    ),

    "Requirements": (
        plm.requirements,
        [
            "Requirement_ID",
            "Product_ID",
            "Requirement_Text",
        ],
    ),

    "Relationships": (
        plm.relationships,
        [
            "Relationship_ID",
            "Source_ID",
            "Target_ID",
        ],
    ),

    "Engineering Changes": (
        plm.changes,
        [
            "ECR_ID",
            "Primary_Component_ID",
            "Change_Title",
        ],
    ),

    "Change Impacts": (
        plm.change_impacts,
        [
            "Impact_ID",
            "ECR_ID",
            "Affected_Entity_ID",
        ],
    ),

    "Manufacturing": (
        plm.manufacturing,
        [
            "Manufacturing_ID",
            "Component_ID",
        ],
    ),

    "Validation": (
        plm.validation,
        [
            "Test_ID",
            "Requirement_ID",
        ],
    ),

    "Service": (
        plm.service,
        [
            "Service_ID",
            "Component_ID",
        ],
    ),

    "Suppliers": (
        plm.suppliers,
        [
            "Supplier_ID",
            "Primary_Component_ID",
        ],
    ),
}


for dataset_name, dataset_info in critical_columns.items():

    dataframe = dataset_info[0]
    columns = dataset_info[1]

    for column in columns:

        if column not in dataframe.columns:
            continue

        missing_count = dataframe[column].isna().sum()

        if missing_count > 0:
            warnings.append(
                f"{dataset_name}: {missing_count} missing "
                f"value(s) in {column}"
            )


# ==========================================================
# 15. ECR IMPACT COVERAGE CHECK
# ==========================================================

for ecr_id in sorted(ecr_ids):

    impacts = plm.change_impacts[
        plm.change_impacts["ECR_ID"] == ecr_id
    ]

    if impacts.empty:

        # Intentional discovery cases are not errors or warnings.
        if ecr_id in discovery_ecrs:
            continue

        warnings.append(
            f"{ecr_id} has no change-impact records."
        )


# ==========================================================
# 16. PRIMARY COMPONENT IMPACT CHECK
# ==========================================================

for _, row in plm.changes.iterrows():

    ecr_id = str(row["ECR_ID"])
    primary_component = str(row["Primary_Component_ID"])

    # Discovery cases intentionally have no impact labels.
    if ecr_id in discovery_ecrs:
        continue

    matching_impacts = plm.change_impacts[
        (
            plm.change_impacts["ECR_ID"] == ecr_id
        )
        &
        (
            plm.change_impacts["Affected_Entity_ID"]
            == primary_component
        )
    ]

    if matching_impacts.empty:
        warnings.append(
            f"{ecr_id} primary component "
            f"{primary_component} is not listed "
            f"in its impact records."
        )


# ==========================================================
# 17. DISCOVERY ECR STATUS CHECK
# ==========================================================

for ecr_id in sorted(discovery_ecrs):

    if ecr_id not in ecr_ids:
        errors.append(
            f"Configured discovery case {ecr_id} "
            f"does not exist in Engineering Changes."
        )
        continue

    discovery_impacts = plm.change_impacts[
        plm.change_impacts["ECR_ID"] == ecr_id
    ]

    if not discovery_impacts.empty:
        warnings.append(
            f"{ecr_id} is configured as a discovery case "
            f"but currently contains "
            f"{len(discovery_impacts)} production impact record(s)."
        )


# ==========================================================
# REPORT
# ==========================================================

print("\nAUREON PLM DATA QUALITY AUDIT")
print("=" * 90)


print("\nDATASET COUNTS")
print("-" * 90)

print("Components:", len(plm.components))
print("Requirements:", len(plm.requirements))
print("Relationships:", len(plm.relationships))
print("Engineering Changes:", len(plm.changes))
print("Change Impact Links:", len(plm.change_impacts))
print("Manufacturing Records:", len(plm.manufacturing))
print("Validation Tests:", len(plm.validation))
print("Service Records:", len(plm.service))
print("Supplier Records:", len(plm.suppliers))


print("\nDISCOVERY ECRs")
print("-" * 90)

for ecr_id in sorted(discovery_ecrs):
    print(ecr_id)


print("\nERRORS")
print("-" * 90)

if errors:

    for error in errors:
        print("ERROR:", error)

else:
    print("No integrity errors detected.")


print("\nWARNINGS")
print("-" * 90)

if warnings:

    for warning in warnings:
        print("WARNING:", warning)

else:
    print("No warnings detected.")


print("\nFINAL RESULT")
print("=" * 90)

if errors:

    print(
        f"FAIL - {len(errors)} integrity issue(s) detected."
    )

else:

    print(
        "PASS - Core PLM cross-file integrity checks passed."
    )

    if warnings:

        print(
            f"NOTE - {len(warnings)} warning(s) detected "
            f"for review."
        )

    else:

        print(
            "Dataset is ready for deterministic PLM "
            "baseline freeze."
        )