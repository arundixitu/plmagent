from collections import defaultdict


def build_evidence_report(plm, change_id):
    """
    Build a structured evidence report for an engineering change.
    """

    analysis = plm.analyze_change(change_id)

    if analysis is None:
        return None

    grouped_impacts = defaultdict(list)

    for item in analysis["enriched_impacts"]:

        impact = item["impact"]
        details = item["entity_details"]

        dimension = impact["Impact_Dimension"]

        grouped_impacts[dimension].append(
            {
                "entity_id": impact["Affected_Entity_ID"],
                "propagation_level": impact["Propagation_Level"],
                "status": impact["Status"],
                "rationale": impact["Rationale"],
                "recommended_action": impact["Recommended_Action"],
                "details": details
            }
        )

    report = {
        "change": analysis["change"],
        "primary_component": analysis["primary_component"],
        "explicit_relationships": analysis["related_components"],
        "impacts_by_dimension": dict(grouped_impacts),
        "total_impacts": len(analysis["enriched_impacts"])
    }

    return report