import pandas as pd


files = {
    "BOM": "data/raw/07_BOM.xlsx",
    "DESIGN SPECIFICATIONS": "data/raw/08_Design_Specifications.xlsx",
    "DOCUMENT CONTROL": "data/raw/16_Document_Control.xlsx",
    "RELATIONSHIP MATRIX": "data/raw/18_Relationship_Matrix.xlsx",
}


for name, file_path in files.items():

    data = pd.read_excel(file_path)

    print("\n")
    print("=" * 100)
    print(name)
    print("=" * 100)

    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    print("\nCOLUMN NAMES")
    print("-" * 100)

    for column in data.columns:
        print(column)

    print("\nFIRST 10 RECORDS")
    print("-" * 100)

    print(
        data.head(10).to_string(index=False)
    )