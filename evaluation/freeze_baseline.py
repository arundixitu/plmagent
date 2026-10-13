from pathlib import Path
from datetime import datetime
import hashlib
import json


DATA_FOLDER = Path("data/raw")
OUTPUT_FILE = Path("evaluation/Aureon_PLM_World_V1_manifest.json")


def calculate_sha256(file_path):
    """
    Calculate SHA-256 fingerprint for a file.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(65536)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


excel_files = sorted(DATA_FOLDER.glob("*.xlsx"))


manifest = {
    "baseline_name": "Aureon E-Move X1 PLM World V1",
    "status": "FROZEN",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "production_ground_truth_policy": (
        "99_Ground_Truth_Impact_Matrix.xlsx is evaluator-only "
        "and must never be exposed to production RAG, tools, "
        "agent prompts, or runtime reasoning."
    ),
    "discovery_ecrs": [
        "ECR-002",
        "ECR-003",
        "ECR-010",
        "ECR-011",
        "ECR-014"
    ],
    "files": []
}


for file_path in excel_files:

    manifest["files"].append(
        {
            "file_name": file_path.name,
            "sha256": calculate_sha256(file_path),
            "size_bytes": file_path.stat().st_size
        }
    )


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        manifest,
        file,
        indent=4
    )


print("\nAUREON PLM WORLD V1 BASELINE FREEZE")
print("=" * 90)

print("Baseline:", manifest["baseline_name"])
print("Status:", manifest["status"])
print("Files recorded:", len(manifest["files"]))

print("\nFILE FINGERPRINTS")
print("-" * 90)

for item in manifest["files"]:
    print(
        item["file_name"],
        "|",
        item["sha256"][:16] + "..."
    )


print("\nManifest created:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "99_Ground_Truth_Impact_Matrix.xlsx remains evaluator-only."
)

print("\nBASELINE FREEZE COMPLETE.")