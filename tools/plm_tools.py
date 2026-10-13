import sys
from pathlib import Path


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from plm.data_engine import PLMDataEngine


class PLMTools:
    """
    Structured deterministic tools for the Aureon PLM agent.
    """

    def __init__(self):
        self.plm = PLMDataEngine()


    def get_component(self, component_id):
        """
        Return exact component information.
        """
        return self.plm.get_component(component_id)


    def get_product_requirements(self, product_id):
        """
        Return all requirements for a product.
        """
        return self.plm.get_product_requirements(product_id)


    def search_requirements(self, keyword):
        """
        Search requirements using a keyword.
        """
        return self.plm.search_requirements(keyword)


    def get_related_components(self, component_id):
        """
        Return explicit product relationships
        connected to a component.
        """
        return self.plm.get_related_components(component_id)


    def get_change(self, ecr_id):
        """
        Return the engineering change record.
        """
        return self.plm.get_change(ecr_id)


    def get_change_impacts(self, ecr_id):
        """
        Return historical impact records for an ECR.

        IMPORTANT:
        This tool must not be exposed for the active
        benchmark ECR during controlled evaluation.
        """
        return self.plm.get_change_impacts(ecr_id)


    def get_requirement(self, requirement_id):
        """
        Return one requirement record.
        """
        return self.plm.get_requirement(requirement_id)


    def get_validation_test(self, test_id):
        """
        Return one validation-test record.
        """
        return self.plm.get_validation_test(test_id)


    def get_manufacturing_record(self, manufacturing_id):
        """
        Return one manufacturing record.
        """
        return self.plm.get_manufacturing_record(
            manufacturing_id
        )


    def get_service_record(self, service_id):
        """
        Return one service record.
        """
        return self.plm.get_service_record(service_id)


    def get_supplier(self, supplier_id):
        """
        Return one supplier record.
        """
        return self.plm.get_supplier(supplier_id)


    def get_entity_details(self, entity_id):
        """
        Resolve an entity ID to its detailed PLM record.
        """
        return self.plm.get_entity_details(entity_id)


    def analyze_change(self, ecr_id):
        """
        Build deterministic structured evidence
        for an engineering change.
        """
        return self.plm.analyze_change(ecr_id)
    
    def get_bom(self, product_id):
        """
        Return the product BOM.
        """
        return self.plm.get_bom(product_id)


    def get_bom_for_component(self, component_id):
        """
        Return BOM line(s) for a component.
        """
        return self.plm.get_bom_for_component(
            component_id
        )


    def get_design_specs(self, component_id):
        """
        Return design specifications for a component.
        """
        return self.plm.get_design_specs(
            component_id
        )


    def get_documents(self, object_id):
        """
        Return controlled documents associated
        with a PLM object.
        """
        return self.plm.get_documents(
            object_id
        )


    def trace_entity(self, entity_id):
        """
        Trace an entity through explicit lifecycle
        relationships.
        """
        return self.plm.trace_entity(
            entity_id
        )