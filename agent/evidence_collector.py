from pathlib import Path
import pandas as pd


DATA_FOLDER = Path("data/raw")


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _records_for_entity(df, column, entity_id):
    """
    Return dataframe records where column matches entity_id.
    """

    if column not in df.columns:
        return []

    result = df[
        df[column].astype(str) == str(entity_id)
    ]

    return result.to_dict(
        orient="records"
    )


def _unique_records(records, key_field):
    """
    Remove duplicate dictionaries using a PLM ID field.
    """

    unique = {}

    for record in records:

        key = record.get(key_field)

        if key is not None:
            unique[str(key)] = record

    return list(
        unique.values()
    )


# ==========================================================
# MAIN EVIDENCE COLLECTOR
# ==========================================================

def collect_structured_evidence(
    tools,
    ecr_id,
    benchmark_mode=False
):
    """
    Build an enriched deterministic PLM evidence package.

    V1.2 includes:
    - primary component evidence
    - direct product relationships
    - related components
    - specifications
    - requirements
    - two-hop lifecycle traversal
    - validation discovery
    - BOM
    - suppliers
    - manufacturing
    - service
    - field failures
    - controlled documents

    In benchmark mode, service and field records explicitly
    tied to the ECR under evaluation are withheld.
    """

    # ======================================================
    # 1. ENGINEERING CHANGE
    # ======================================================

    change = tools.get_change(
        ecr_id
    )

    if change is None:
        return None


    primary_component_id = str(
        change["Primary_Component_ID"]
    )


    # ======================================================
    # 2. PRIMARY COMPONENT
    # ======================================================

    primary_component = (
        tools.get_component(
            primary_component_id
        )
    )


    # ======================================================
    # 3. PRODUCT RELATIONSHIPS
    # ======================================================

    product_relationships = (
        tools.get_related_components(
            primary_component_id
        )
    )


    # ======================================================
    # 4. RELATED COMPONENT IDS
    # ======================================================

    component_ids = set(
        tools.plm.components[
            "Component_ID"
        ]
        .dropna()
        .astype(str)
    )


    related_component_ids = set()


    for relationship in product_relationships:

        source = str(
            relationship["Source_ID"]
        )

        target = str(
            relationship["Target_ID"]
        )


        if source == primary_component_id:

            candidate = target

        else:

            candidate = source


        if candidate in component_ids:

            related_component_ids.add(
                candidate
            )


    # ======================================================
    # 5. RELATED COMPONENT DETAILS
    # ======================================================

    related_components = []


    for component_id in sorted(
        related_component_ids
    ):

        component = (
            tools.get_component(
                component_id
            )
        )

        if component is not None:

            related_components.append(
                component
            )


    # ======================================================
    # 6. PRIMARY COMPONENT DESIGN SPECIFICATIONS
    # ======================================================

    design_specifications = (
        tools.get_design_specs(
            primary_component_id
        )
    )


    # ======================================================
    # 7. REQUIREMENT DISCOVERY
    # ======================================================

    requirement_ids = set()


    # ------------------------------------------------------
    # 7A. Requirements linked through specifications
    # ------------------------------------------------------

    for specification in design_specifications:

        requirement_id = (
            specification.get(
                "Requirement_ID"
            )
        )

        if (
            requirement_id is not None
            and pd.notna(requirement_id)
        ):

            requirement_ids.add(
                str(requirement_id)
            )


    # ======================================================
    # 8. FIRST-HOP LIFECYCLE TRACE
    # ======================================================

    primary_trace = (
        tools.trace_entity(
            primary_component_id
        )
    )


    first_hop_relationships = (
        primary_trace.get(
            "relationships",
            []
        )
    )


    first_hop_entity_ids = set()


    for relationship in first_hop_relationships:

        source = str(
            relationship["Source_ID"]
        )

        target = str(
            relationship["Target_ID"]
        )


        if source == primary_component_id:

            neighbor = target

        elif target == primary_component_id:

            neighbor = source

        else:

            continue


        first_hop_entity_ids.add(
            neighbor
        )


        if neighbor.startswith(
            "REQ-"
        ):

            requirement_ids.add(
                neighbor
            )


    # ======================================================
    # 9. SECOND-HOP LIFECYCLE TRACE
    # ======================================================

    second_hop_relationships = []

    second_hop_entity_ids = set()


    for entity_id in sorted(
        first_hop_entity_ids
    ):

        trace = tools.trace_entity(
            entity_id
        )

        relationships = trace.get(
            "relationships",
            []
        )


        for relationship in relationships:

            second_hop_relationships.append(
                relationship
            )


            source = str(
                relationship["Source_ID"]
            )

            target = str(
                relationship["Target_ID"]
            )


            if source == entity_id:

                neighbor = target

            elif target == entity_id:

                neighbor = source

            else:

                continue


            # Prevent trivial return to root
            if neighbor == primary_component_id:
                continue


            second_hop_entity_ids.add(
                neighbor
            )


            if neighbor.startswith(
                "REQ-"
            ):

                requirement_ids.add(
                    neighbor
                )


    # ======================================================
    # 10. REQUIREMENTS
    # ======================================================

    requirements = []


    for requirement_id in sorted(
        requirement_ids
    ):

        requirement = (
            tools.get_requirement(
                requirement_id
            )
        )

        if requirement is not None:

            requirements.append(
                requirement
            )


    # ======================================================
    # 11. VALIDATION DISCOVERY
    # ======================================================

    validation_records = []

    validation_df = tools.plm.validation


    # ------------------------------------------------------
    # 11A. Tests directly linked to discovered requirements
    # ------------------------------------------------------

    if requirement_ids:

        requirement_test_matches = (
            validation_df[
                validation_df[
                    "Requirement_ID"
                ]
                .astype(str)
                .isin(
                    requirement_ids
                )
            ]
        )

        validation_records.extend(
            requirement_test_matches.to_dict(
                orient="records"
            )
        )


    # ------------------------------------------------------
    # 11B. Tests discovered from lifecycle relationship graph
    # ------------------------------------------------------

    relationship_matrix = (
        tools.plm._load_excel(
            "18_Relationship_Matrix.xlsx"
        )
    )


    lifecycle_candidate_ids = (
        {primary_component_id}
        .union(first_hop_entity_ids)
        .union(second_hop_entity_ids)
        .union(requirement_ids)
    )


    relevant_relationship_rows = (
        relationship_matrix[
            relationship_matrix[
                "Source_ID"
            ]
            .astype(str)
            .isin(
                lifecycle_candidate_ids
            )
            |
            relationship_matrix[
                "Target_ID"
            ]
            .astype(str)
            .isin(
                lifecycle_candidate_ids
            )
        ]
    )


    discovered_test_ids = set()


    for _, relationship in (
        relevant_relationship_rows.iterrows()
    ):

        source_id = str(
            relationship["Source_ID"]
        )

        target_id = str(
            relationship["Target_ID"]
        )


        if source_id.startswith(
            "TEST-"
        ):

            discovered_test_ids.add(
                source_id
            )


        if target_id.startswith(
            "TEST-"
        ):

            discovered_test_ids.add(
                target_id
            )


    for test_id in sorted(
        discovered_test_ids
    ):

        test = tools.get_validation_test(
            test_id
        )

        if test is not None:

            validation_records.append(
                test
            )


    validation_records = (
        _unique_records(
            validation_records,
            "Test_ID"
        )
    )


    # ======================================================
    # 12. BOM
    # ======================================================

    bom_records = (
        tools.get_bom_for_component(
            primary_component_id
        )
    )


    # ======================================================
    # 13. SUPPLIERS
    # ======================================================

    supplier_records = (
        _records_for_entity(
            tools.plm.suppliers,
            "Primary_Component_ID",
            primary_component_id
        )
    )


    # ======================================================
    # 14. MANUFACTURING
    # ======================================================

    manufacturing_records = (
        _records_for_entity(
            tools.plm.manufacturing,
            "Component_ID",
            primary_component_id
        )
    )


    # ======================================================
    # 15. SERVICE RECORDS
    # ======================================================

    service_df = (
        tools.plm.service.copy()
    )


    service_matches = (
        service_df[
            service_df[
                "Component_ID"
            ].astype(str)
            == primary_component_id
        ].copy()
    )


    if benchmark_mode:

        service_matches = (
            service_matches[
                service_matches[
                    "Related_ECR"
                ]
                .astype(str)
                != ecr_id
            ]
        )


    service_records = (
        service_matches.to_dict(
            orient="records"
        )
    )


    # ======================================================
    # 16. FIELD FAILURE RECORDS
    # ======================================================

    field_df = pd.read_excel(
        DATA_FOLDER /
        "15_Field_Failure_Records.xlsx"
    )


    field_matches = (
        field_df[
            field_df[
                "Component_ID"
            ].astype(str)
            == primary_component_id
        ].copy()
    )


    if benchmark_mode:

        field_matches = (
            field_matches[
                field_matches[
                    "Related_ECR"
                ]
                .astype(str)
                != ecr_id
            ]
        )


    field_failure_records = (
        field_matches.to_dict(
            orient="records"
        )
    )


    # ======================================================
    # 17. CONTROLLED DOCUMENTS
    # ======================================================

    controlled_documents = (
        tools.get_documents(
            primary_component_id
        )
    )


    # ======================================================
    # 18. RELATED COMPONENT SUPPORTING EVIDENCE
    # ======================================================

    related_component_evidence = []


    for component_id in sorted(
        related_component_ids
    ):

        component_data = {
            "component":
                tools.get_component(
                    component_id
                ),

            "specifications":
                tools.get_design_specs(
                    component_id
                ),

            "bom":
                tools.get_bom_for_component(
                    component_id
                ),

            "documents":
                tools.get_documents(
                    component_id
                ),
        }


        related_component_evidence.append(
            component_data
        )


    # ======================================================
    # 19. LIFECYCLE TRACE PACKAGE
    # ======================================================

    lifecycle_trace = {

        "root_entity":
            primary_component_id,

        "first_hop_relationships":
            first_hop_relationships,

        "first_hop_entities":
            sorted(
                first_hop_entity_ids
            ),

        "second_hop_relationships":
            second_hop_relationships,

        "second_hop_entities":
            sorted(
                second_hop_entity_ids
            ),
    }


    # ======================================================
    # 20. FINAL EVIDENCE PACKAGE
    # ======================================================

    evidence = {

        "change":
            change,

        "primary_component":
            primary_component,

        "product_relationships":
            product_relationships,

        "related_components":
            related_components,

        "related_component_evidence":
            related_component_evidence,

        "design_specifications":
            design_specifications,

        "requirements":
            requirements,

        "validation_records":
            validation_records,

        "bom_records":
            bom_records,

        "supplier_records":
            supplier_records,

        "manufacturing_records":
            manufacturing_records,

        "service_records":
            service_records,

        "field_failure_records":
            field_failure_records,

        "controlled_documents":
            controlled_documents,

        "lifecycle_trace":
            lifecycle_trace,
    }


    return evidence