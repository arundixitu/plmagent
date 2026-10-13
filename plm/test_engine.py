from data_engine import PLMDataEngine
from evidence_builder import build_evidence_report


plm = PLMDataEngine()

report = build_evidence_report(plm, "ECR-001")


print("\nAUREON PLM ENGINEERING EVIDENCE REPORT")
print("=" * 90)


print("\nCHANGE")
print("-" * 90)

change = report["change"]

print("ECR:", change["ECR_ID"])
print("Title:", change["Change_Title"])
print("Description:", change["Change_Description"])
print("Risk:", change["Risk"])
print("Status:", change["Status"])


print("\nPRIMARY COMPONENT")
print("-" * 90)

component = report["primary_component"]

print(
    component["Component_ID"],
    "-",
    component["Component_Name"],
    "-",
    component["Baseline_Description"]
)


print("\nEXPLICIT RELATIONSHIPS")
print("-" * 90)

for relationship in report["explicit_relationships"]:
    print(
        relationship["Source_ID"],
        "--",
        relationship["Relationship_Type"],
        "-->",
        relationship["Target_ID"]
    )


print("\nIMPACTS BY LIFECYCLE DIMENSION")
print("=" * 90)

for dimension, impacts in report["impacts_by_dimension"].items():

    print(f"\n{dimension.upper()}")
    print("-" * 90)

    for item in impacts:

        print(
            item["entity_id"],
            "|",
            item["propagation_level"]
        )

        details = item["details"]

        if details:

            if "Component_Name" in details:
                print("   Name:", details["Component_Name"])

            elif "Requirement_Text" in details:
                print(
                    "   Requirement:",
                    details["Requirement_Text"],
                    "| Baseline:",
                    details["Baseline_Value"]
                )

            elif "Test_Name" in details:
                print(
                    "   Test:",
                    details["Test_Name"],
                    "| Status:",
                    details["Status"]
                )

            elif "Process_Name" in details:
                print(
                    "   Manufacturing Process:",
                    details["Process_Name"]
                )

            elif "Issue" in details:
                print(
                    "   Service Issue:",
                    details["Issue"],
                    "| Action:",
                    details["Service_Action"]
                )


print("\nSUMMARY")
print("=" * 90)

for dimension, impacts in report["impacts_by_dimension"].items():
    print(f"{dimension}: {len(impacts)}")

print("\nTOTAL IMPACTS:", report["total_impacts"])