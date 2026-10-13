from pathlib import Path
import pandas as pd


class PLMDataEngine:
    """
    Data access layer for the Aureon E-Move X1 PLM World.
    """

    def __init__(self, data_folder="data/raw"):
        self.data_folder = Path(data_folder)

        self.components = self._load_excel("04_Component_Master.xlsx")
        self.relationships = self._load_excel("05_Product_Relationships.xlsx")
        self.requirements = self._load_excel("06_Product_Requirements.xlsx")
        self.bom = self._load_excel("07_BOM.xlsx")
        self.suppliers = self._load_excel("09_Supplier_Records.xlsx")
        self.design_specs = self._load_excel("08_Design_Specifications.xlsx")
        self.changes = self._load_excel("10_Engineering_Changes.xlsx")
        self.change_impacts = self._load_excel("11_Change_Impact_Links.xlsx")
        self.manufacturing = self._load_excel("12_Manufacturing_Records.xlsx")
        self.validation = self._load_excel("13_Test_Validation_Records.xlsx")
        self.service = self._load_excel("14_Service_Records.xlsx")

        print("Aureon PLM Data Engine initialized successfully.")

    def _load_excel(self, filename):
        """
        Load an Excel file from the PLM data folder.
        """
        file_path = self.data_folder / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"PLM dataset not found: {file_path}"
            )

        return pd.read_excel(file_path)

    def get_component(self, component_id):
        """
        Retrieve a component using its Component_ID.
        """
        result = self.components[
            self.components["Component_ID"] == component_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_product_requirements(self, product_id):
        """
        Retrieve all requirements belonging to a product.
        """
        result = self.requirements[
            self.requirements["Product_ID"] == product_id
        ]

        return result.to_dict(orient="records")

    def search_requirements(self, keyword):
        """
        Search requirements using a keyword.
        """
        result = self.requirements[
            self.requirements["Requirement_Text"]
            .str.contains(keyword, case=False, na=False)
        ]

        return result.to_dict(orient="records")

    def get_related_components(self, component_id):
        """
        Retrieve explicit component relationships.
        """
        source_matches = self.relationships[
            self.relationships["Source_ID"] == component_id
        ]

        target_matches = self.relationships[
            self.relationships["Target_ID"] == component_id
        ]

        result = pd.concat(
            [source_matches, target_matches],
            ignore_index=True
        )

        return result.to_dict(orient="records")

    def get_change(self, change_id):
        """
        Retrieve an engineering change request.
        """
        result = self.changes[
            self.changes["ECR_ID"] == change_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_change_impacts(self, change_id):
        """
        Retrieve all impact links associated with an engineering change.
        """
        result = self.change_impacts[
            self.change_impacts["ECR_ID"] == change_id
        ]

        return result.to_dict(orient="records")

    def analyze_change(self, change_id):
        """
        Build a structured PLM change-analysis package
        with enriched evidence.
        """
        change = self.get_change(change_id)

        if change is None:
            return None

        primary_component_id = change["Primary_Component_ID"]

        impact_links = self.get_change_impacts(change_id)

        enriched_impacts = []

        for impact in impact_links:
            entity_id = impact["Affected_Entity_ID"]

            enriched_impacts.append(
                {
                    "impact": impact,
                    "entity_details": self.get_entity_details(entity_id)
                }
            )

        analysis = {
            "change": change,
            "primary_component": self.get_component(
                primary_component_id
            ),
            "related_components": self.get_related_components(
                primary_component_id
            ),
            "impact_links": impact_links,
            "enriched_impacts": enriched_impacts
        }

        return analysis

    def get_requirement(self, requirement_id):
        """
        Retrieve a requirement using Requirement_ID.
        """
        result = self.requirements[
            self.requirements["Requirement_ID"] == requirement_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_validation_test(self, test_id):
        """
        Retrieve a validation test using Test_ID.
        """
        result = self.validation[
            self.validation["Test_ID"] == test_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_manufacturing_record(self, manufacturing_id):
        """
        Retrieve a manufacturing record using Manufacturing_ID.
        """
        result = self.manufacturing[
            self.manufacturing["Manufacturing_ID"] == manufacturing_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_service_record(self, service_id):
        """
        Retrieve a service record using Service_ID.
        """
        result = self.service[
            self.service["Service_ID"] == service_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def get_entity_details(self, entity_id):
        """
        Retrieve details for an impacted PLM entity
        based on its ID prefix.
        """

        if entity_id.startswith("REQ-"):
            return self.get_requirement(entity_id)

        if entity_id.startswith("TEST-"):
            return self.get_validation_test(entity_id)

        if entity_id.startswith("MFG-"):
            return self.get_manufacturing_record(entity_id)

        if entity_id.startswith("SRV-"):
            return self.get_service_record(entity_id)

        if entity_id.startswith("SUP-"):
            return self.get_supplier(entity_id)

        component = self.get_component(entity_id)

        if component is not None:
            return component

        return None

    def get_supplier(self, supplier_id):
        """
        Retrieve a supplier using Supplier_ID.
        """
        result = self.suppliers[
            self.suppliers["Supplier_ID"] == supplier_id
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()
    
    def get_bom(self, product_id):
        """
        Retrieve the complete BOM for a product.
        """
        result = self.bom[
            self.bom["Product_ID"] == product_id
        ]

        return result.to_dict(orient="records")


    def get_bom_for_component(self, component_id):
        """
        Retrieve BOM line(s) associated with a component.
        """
        result = self.bom[
            self.bom["Component_ID"] == component_id
        ]

        return result.to_dict(orient="records")


    def get_design_specs(self, component_id):
        """
        Retrieve design specifications associated with a component.
        """
        result = self.design_specs[
            self.design_specs["Component_ID"] == component_id
        ]

        return result.to_dict(orient="records")


    def get_documents(self, object_id):
        """
        Retrieve controlled documents associated with a PLM object.
        """

        if not hasattr(self, "documents"):
            self.documents = self._load_excel(
                "16_Document_Control.xlsx"
            )

        result = self.documents[
            self.documents["Object_ID"] == object_id
        ]

        return result.to_dict(orient="records")


    def get_relationship_matrix_links(self, entity_id):
        """
        Retrieve explicit lifecycle traceability links
        from the relationship matrix.
        """

        if not hasattr(self, "relationship_matrix"):
            self.relationship_matrix = self._load_excel(
                "18_Relationship_Matrix.xlsx"
            )

        source_matches = self.relationship_matrix[
            self.relationship_matrix["Source_ID"] == entity_id
        ]

        target_matches = self.relationship_matrix[
            self.relationship_matrix["Target_ID"] == entity_id
        ]

        result = pd.concat(
            [
                source_matches,
                target_matches
            ],
            ignore_index=True
        )

        return result.to_dict(
            orient="records"
        )


    def trace_entity(self, entity_id):
        """
        Build a structured lifecycle trace for an entity.
        """

        links = self.get_relationship_matrix_links(
            entity_id
        )

        trace = {
            "entity_id": entity_id,
            "entity_details": self.get_entity_details(
                entity_id
            ),
            "relationships": links
        }

        return trace