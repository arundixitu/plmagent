from plm_tools import PLMTools


tools = PLMTools()


print("\nAUREON ADVANCED PLM TOOL TEST")
print("=" * 90)


# ==========================================================
# 1. BOM
# ==========================================================

print("\n1. PRODUCT BOM")
print("-" * 90)

bom = tools.get_bom("PROD-X1")

print("BOM lines:", len(bom))

for row in bom[:5]:
    print(
        row["BOM_Line_ID"],
        "-",
        row["Component_ID"],
        "-",
        row["Component_Name"],
        "-",
        row["Revision"]
    )


# ==========================================================
# 2. BATTERY BOM LINE
# ==========================================================

print("\n2. BATTERY BOM LINE")
print("-" * 90)

battery_bom = tools.get_bom_for_component(
    "BAT-001"
)

for row in battery_bom:
    print(row)


# ==========================================================
# 3. BATTERY DESIGN SPECIFICATIONS
# ==========================================================

print("\n3. BATTERY DESIGN SPECIFICATIONS")
print("-" * 90)

specs = tools.get_design_specs(
    "BAT-001"
)

for row in specs:
    print(
        row["Spec_ID"],
        "-",
        row["Parameter"],
        "=",
        row["Value"],
        "- Requirement:",
        row["Requirement_ID"]
    )


# ==========================================================
# 4. BATTERY DOCUMENTS
# ==========================================================

print("\n4. BATTERY DOCUMENTS")
print("-" * 90)

documents = tools.get_documents(
    "BAT-001"
)

for row in documents:
    print(
        row["Document_ID"],
        "-",
        row["Document_Title"],
        "-",
        row["Revision"]
    )


# ==========================================================
# 5. REQUIREMENT TRACE
# ==========================================================

print("\n5. TRACE REQ-014")
print("-" * 90)

trace = tools.trace_entity(
    "REQ-014"
)

print(
    "Entity:",
    trace["entity_id"]
)

print(
    "Details:",
    trace["entity_details"]
)

print("\nRelationships:")

for relationship in trace["relationships"]:

    print(
        relationship["Source_ID"],
        "--",
        relationship["Relationship_Type"],
        "-->",
        relationship["Target_ID"]
    )


# ==========================================================
# 6. BATTERY TRACE
# ==========================================================

print("\n6. TRACE BAT-001")
print("-" * 90)

trace = tools.trace_entity(
    "BAT-001"
)

print(
    "Entity:",
    trace["entity_id"]
)

print("\nRelationships:")

for relationship in trace["relationships"]:

    print(
        relationship["Source_ID"],
        "--",
        relationship["Relationship_Type"],
        "-->",
        relationship["Target_ID"]
    )


print("\nADVANCED PLM TOOL TEST COMPLETE.")