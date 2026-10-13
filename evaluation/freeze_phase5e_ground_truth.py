from pathlib import Path
import hashlib
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

source = (
    PROJECT_ROOT
    / "evaluation"
    / "phase5e"
    / "Phase5E_Unseen_Expert_Labeling_Template.xlsx"
)

freeze_dir = (
    PROJECT_ROOT
    / "evaluation"
    / "phase5e"
    / "frozen_ground_truth"
)

freeze_dir.mkdir(parents=True, exist_ok=True)

frozen_file = (
    freeze_dir
    / "Phase5E_Frozen_Ground_Truth.xlsx"
)

if not source.exists():
    raise FileNotFoundError(
        f"Could not find source file:\n{source}"
    )

# Make frozen copy
shutil.copy2(source, frozen_file)

# Calculate SHA-256
sha256 = hashlib.sha256()

with open(frozen_file, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        sha256.update(chunk)

hash_value = sha256.hexdigest()

hash_file = (
    freeze_dir
    / "Phase5E_Frozen_Ground_Truth_SHA256.txt"
)

with open(hash_file, "w", encoding="utf-8") as f:
    f.write("Aureon Phase 5E Frozen Ground Truth\n")
    f.write("=" * 50 + "\n")
    f.write(f"File: {frozen_file.name}\n")
    f.write(f"SHA-256: {hash_value}\n")

print("\nPHASE 5E GROUND TRUTH FROZEN SUCCESSFULLY")
print("=" * 60)

print("\nFrozen file:")
print(frozen_file)

print("\nSHA-256:")
print(hash_value)

print("\nHash record:")
print(hash_file)

print("\nIMPORTANT:")
print("Do not edit the frozen ground-truth file from this point onward.")
print("The prospective benchmark is now ready for agent evaluation.")