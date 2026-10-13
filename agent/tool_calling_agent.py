import os
import sys
import json
import math
import time
import re
from datetime import datetime
from pathlib import Path


# ==========================================================
# WINDOWS DLL INITIALIZATION
# ==========================================================

# PyTorch must load before pandas / chromadb on this machine.
import torch


from dotenv import load_dotenv
from openai import OpenAI


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Keep RAG import before PLMTools because of the Windows
# native-library loading behaviour observed on this machine.
from rag.retriever import AureonRetriever
from tools.plm_tools import PLMTools

import pandas as pd


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna"
)

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in .env"
    )


client = OpenAI(
    api_key=API_KEY
)


# ==========================================================
# AGENT LIMITS
# ==========================================================

MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 30


# ==========================================================
# PREDICTION OUTPUT FOLDER
# ==========================================================

PREDICTIONS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "predictions"
)

PREDICTIONS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INITIALIZE PLM CAPABILITIES
# ==========================================================

plm_tools = PLMTools()

retriever = AureonRetriever()


# ==========================================================
# DATA PATH
# ==========================================================

DATA_FOLDER = PROJECT_ROOT / "data" / "raw"

FIELD_FAILURE_FILE = (
    DATA_FOLDER
    / "15_Field_Failure_Records.xlsx"
)

field_failures_df = pd.read_excel(
    FIELD_FAILURE_FILE
)


# ==========================================================
# AGENT INSTRUCTIONS
# ==========================================================

AGENT_INSTRUCTIONS = """
You are the Aureon Autonomous PLM Engineering Agent.

You analyse engineering changes for the fictional
Aureon E-Move X1 product lifecycle.

You have access to structured PLM tools, lifecycle
traceability tools and historical semantic retrieval.

Your responsibility is not to call every available tool.
Your responsibility is to gather sufficient, relevant
engineering evidence efficiently.

============================================================
CORE WORKFLOW
============================================================

Internally follow:

UNDERSTAND GOAL
→ IDENTIFY EVIDENCE GAPS
→ SELECT TOOL
→ INSPECT RESULT
→ FOLLOW RELEVANT TRACEABILITY
→ CHECK WHETHER MORE EVIDENCE IS NECESSARY
→ VALIDATE CLAIMS
→ REPORT

After retrieving the active engineering change using
get_change, normally inspect its lifecycle trace using
trace_entity on the active ECR before performing broad
component exploration.

The ECR lifecycle trace is a high-value source for explicit
validation, documentation, field, service and change-control
relationships.

In benchmark mode, the application may filter active-ECR
relationships.

Accept the filtered result.

Do not attempt to reconstruct, recover or infer hidden
active-ECR relationships through indirect searches.

============================================================
TOOL EFFICIENCY RULES
============================================================

Use tools selectively.

Before requesting another tool, ask whether its output is
likely to materially change:

- impact identification;
- engineering confidence;
- risk assessment; or
- recommended action.

Do not retrieve records merely because they exist.

Prefer high-information traceability calls before multiple
low-level record lookups.

For an engineering-change assessment, an ECR lifecycle
trace should normally be inspected before individually
searching many downstream records.

Prefer stopping once the major relevant lifecycle dimensions
have been sufficiently checked.

As a general target, complete a typical engineering change
assessment using approximately 12 to 25 useful tool calls.

The application enforces a hard tool-call budget.

Do not repeat an identical tool request unless new evidence
makes repetition necessary.

============================================================
EVIDENCE HIERARCHY
============================================================

Always distinguish:

1. KNOWN FROM PLM DATA
2. EXPLICIT RELATIONSHIP
3. HISTORICAL EVIDENCE
4. CALCULATED
5. ENGINEERING INFERENCE
6. UNKNOWN / MISSING INFORMATION

Never convert an inference into confirmed PLM evidence.

============================================================
CLAIM DISCIPLINE
============================================================

For important conclusions use:

CLAIM
→ EVIDENCE
→ ENGINEERING REASONING
→ CONSEQUENCE
→ RISK
→ RECOMMENDED ACTION

============================================================
NO HALLUCINATION
============================================================

Never invent:

- dimensions
- mass
- costs
- suppliers
- voltage
- current
- manufacturing processes
- test results
- validation status
- requirement values
- field failures
- service events
- regulatory conclusions
- safety conclusions
- component specifications

If evidence is unavailable, identify it as unknown.

============================================================
CHANGE IMPACT RULE
============================================================

A relationship identifies a possible impact path.

It does NOT automatically prove that a related component
requires modification.

Use:

- confirmed impact
- review candidate
- compatibility verification
- potential impact
- engineering inference

appropriately.

============================================================
IMPACT PROPAGATION
============================================================

Consider:

DIRECT IMPACT

FIRST-ORDER IMPACT

SECOND-ORDER IMPACT

LIFECYCLE IMPACT

Do not propagate indefinitely through the product graph.

Follow only relationships that are engineering-relevant to
the active change.

============================================================
IMPACT DIMENSIONS
============================================================

Consider where relevant:

- requirements
- product architecture
- component design
- mechanical
- electrical
- thermal
- software/control
- BOM/configuration
- suppliers
- manufacturing
- validation
- service
- field performance
- documentation
- product performance
- safety review

============================================================
HISTORICAL RAG RULE
============================================================

Use retrieve_historical_evidence when analogous changes,
service events, field failures or historical impacts may
improve engineering judgement.

Historical similarity is contextual evidence.

It is not proof of:

- identical root cause;
- identical design;
- identical impact; or
- identical solution.

============================================================
VALIDATION RULE
============================================================

Validation of a baseline configuration does not validate a
changed configuration.

============================================================
CALCULATION RULE
============================================================

Straightforward engineering calculations are permitted only
when all required inputs are explicitly available.

Clearly identify:

- inputs;
- assumptions;
- calculation; and
- result.

Never present a calculated value as a stored PLM value.

============================================================
SECURITY / EVALUATION RULE
============================================================

The following evaluator-only information must never be
accessed:

99_Ground_Truth_Impact_Matrix.xlsx

Direct structured impact-label access is also prohibited.

Do not request or attempt to resolve entities beginning with:

IMPACT-

Do not request:

get_change_impacts

or:

analyze_change

Historical impact knowledge may only be accessed indirectly
through benchmark-safe historical retrieval.

============================================================
BENCHMARK MODE
============================================================

During benchmark mode, evidence explicitly belonging to the
active ECR may be withheld.

This may include:

- lifecycle relationships involving the active ECR;
- service records explicitly linked to the active ECR;
- field-failure records explicitly linked to the active ECR;
- historical RAG evidence explicitly belonging to the
  active ECR;
- other records returned indirectly that explicitly identify
  the active ECR.

Do not interpret withheld evidence as evidence that it never
existed.

Do not attempt to recover withheld evidence through generic
entity lookup, relationship traversal, semantic search or
other indirect routes.

============================================================
RELATIONSHIP TABLE RULE
============================================================

Relationship IDs are local to their source tables.

Identical Relationship_ID values in:

05_Product_Relationships.xlsx

and:

18_Relationship_Matrix.xlsx

do not automatically represent a conflict.

============================================================
FINAL OUTPUT
============================================================

When evidence is sufficient, provide:

CHANGE SUMMARY

EVIDENCE CONSULTED

CONFIRMED DIRECT IMPACT

LIKELY FIRST-ORDER IMPACTS

LIKELY SECOND-ORDER / LIFECYCLE IMPACTS

REQUIREMENTS IMPACT

DESIGN / COMPONENT IMPACT

BOM / CONFIGURATION IMPACT

VALIDATION IMPACT

MANUFACTURING / SUPPLIER IMPACT

SERVICE / FIELD IMPACT

DOCUMENTATION IMPACT

HISTORICAL EVIDENCE

UNKNOWN OR MISSING INFORMATION

RISK ASSESSMENT

RECOMMENDED ENGINEERING ACTIONS

PLM LEARNING TAKEAWAY

In EVIDENCE CONSULTED, report only records that actually
contributed to the assessment.

============================================================
MACHINE-READABLE PREDICTION
============================================================

At the very end of the response, after the narrative report,
include exactly one machine-readable JSON block.

The block must begin with:

PREDICTION_JSON

Then provide valid JSON in this format:

{
  "ecr_id": "ECR-001",
  "predicted_impacts": [
    {
      "entity_id": "BAT-001",
      "impact_level": "Direct",
      "confidence": "Confirmed",
      "reason": "Short evidence-grounded reason"
    }
  ]
}

Rules:

- Include only PLM entities that you predict are genuinely
  impacted or require engineering review because of the
  active change.

- Do not include historical records merely because they were
  consulted.

- Do not include unrelated ECRs used only as historical
  context.

- Do not include an entity merely because its record was
  retrieved.

- Allowed impact_level values are:

  Direct
  First-order
  Second-order
  Lifecycle

- Allowed confidence values are:

  Confirmed
  Strong candidate
  Review candidate

- entity_id must be an actual PLM entity ID observed in the
  supplied evidence.

- Do not invent entity IDs.

- Every entity must appear only once.

- If an entity could belong to multiple levels, assign the
  highest-priority level using:

  Direct
  > First-order
  > Second-order
  > Lifecycle

- Do not include IMPACT-* entities.

- The JSON must be valid JSON.

- Do not place any text after the JSON block.
"""


# ==========================================================
# TOOL DEFINITIONS
# ==========================================================

TOOLS = [

    {
        "type": "function",
        "name": "get_change",
        "description":
            "Get an engineering change record using an ECR ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "ecr_id": {
                    "type": "string"
                }
            },
            "required": ["ecr_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_component",
        "description":
            "Get component master-data details.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_related_components",
        "description":
            "Get explicit product relationships involving a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_design_specs",
        "description":
            "Get design specifications for a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_requirement",
        "description":
            "Get one requirement by requirement ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "string"
                }
            },
            "required": ["requirement_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "search_requirements",
        "description":
            "Search requirements by an engineering keyword.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string"
                }
            },
            "required": ["keyword"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_bom_for_component",
        "description":
            "Get BOM records containing a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_supplier_for_component",
        "description":
            "Get supplier records associated with a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_manufacturing_for_component",
        "description":
            "Get manufacturing records associated with a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_validation_test",
        "description":
            "Get a validation test record using a test ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string"
                }
            },
            "required": ["test_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_validation_for_requirement",
        "description":
            "Get validation tests linked to a requirement.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "string"
                }
            },
            "required": ["requirement_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_service_for_component",
        "description":
            "Get service records involving a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_field_failures_for_component",
        "description":
            "Get field-failure records involving a component.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string"
                }
            },
            "required": ["component_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_documents",
        "description":
            "Get controlled documents associated with a PLM object.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string"
                }
            },
            "required": ["object_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "trace_entity",
        "description":
            "Trace lifecycle relationships around a PLM entity.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string"
                }
            },
            "required": ["entity_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_entity_details",
        "description":
            "Resolve a general PLM entity ID. IMPACT entities are prohibited.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string"
                }
            },
            "required": ["entity_id"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "retrieve_historical_evidence",
        "description":
            "Semantic search over permitted historical engineering evidence.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                },
                "top_k": {
                    "type": "integer"
                }
            },
            "required": [
                "query",
                "top_k"
            ],
            "additionalProperties": False
        }
    }
]


# ==========================================================
# JSON CLEANING
# ==========================================================

def clean_json_value(value):

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): clean_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, float):

        if math.isnan(value):
            return None

        return value

    if hasattr(value, "item"):

        try:
            return clean_json_value(
                value.item()
            )

        except Exception:
            pass

    return value


# ==========================================================
# IMPACT-ID SECURITY
# ==========================================================

def is_impact_entity(entity_id):

    return str(
        entity_id
    ).upper().startswith(
        "IMPACT-"
    )


def remove_impact_relationships(
    trace_result
):

    if not isinstance(
        trace_result,
        dict
    ):
        return trace_result

    relationships = trace_result.get(
        "relationships",
        []
    )

    safe_relationships = []

    for relationship in relationships:

        source_id = str(
            relationship.get(
                "Source_ID",
                ""
            )
        )

        target_id = str(
            relationship.get(
                "Target_ID",
                ""
            )
        )

        if (
            is_impact_entity(source_id)
            or is_impact_entity(target_id)
        ):
            continue

        safe_relationships.append(
            relationship
        )

    result = dict(
        trace_result
    )

    result[
        "relationships"
    ] = safe_relationships

    return result


# ==========================================================
# BENCHMARK FILTERING
# ==========================================================

def filter_active_ecr_records(
    records,
    active_ecr
):

    if not active_ecr:
        return records

    active_ecr_upper = str(
        active_ecr
    ).upper()

    filtered = []

    for record in records:

        related_ecr = str(
            record.get(
                "Related_ECR",
                ""
            )
        ).upper()

        if related_ecr != active_ecr_upper:

            filtered.append(
                record
            )

    return filtered


def filter_trace_for_benchmark(
    trace_result,
    active_ecr
):

    if not isinstance(
        trace_result,
        dict
    ):
        return trace_result

    relationships = trace_result.get(
        "relationships",
        []
    )

    safe_relationships = []

    active_ecr_upper = (
        str(active_ecr).upper()
        if active_ecr
        else None
    )

    for relationship in relationships:

        source_id = str(
            relationship.get(
                "Source_ID",
                ""
            )
        ).upper()

        target_id = str(
            relationship.get(
                "Target_ID",
                ""
            )
        ).upper()

        if (
            is_impact_entity(source_id)
            or is_impact_entity(target_id)
        ):
            continue

        if active_ecr_upper:

            if (
                source_id == active_ecr_upper
                or target_id == active_ecr_upper
            ):
                continue

        safe_relationships.append(
            relationship
        )

    result = dict(
        trace_result
    )

    result[
        "relationships"
    ] = safe_relationships

    return result


# ==========================================================
# UNIVERSAL BENCHMARK SANITIZER
# ==========================================================

def sanitize_benchmark_result(
    value,
    active_ecr
):

    if not active_ecr:
        return value

    active_ecr = str(
        active_ecr
    ).upper()


    if isinstance(value, list):

        safe_items = []

        for item in value:

            sanitized = (
                sanitize_benchmark_result(
                    item,
                    active_ecr
                )
            )

            if (
                isinstance(
                    sanitized,
                    dict
                )
                and sanitized.get(
                    "_benchmark_withheld"
                )
            ):
                continue

            safe_items.append(
                sanitized
            )

        return safe_items


    if isinstance(value, dict):

        keys = {
            str(key).lower(): key
            for key in value.keys()
        }


        # --------------------------------------------------
        # RECORD EXPLICITLY LINKED TO ACTIVE ECR
        # --------------------------------------------------

        if "related_ecr" in keys:

            original_key = keys[
                "related_ecr"
            ]

            related_ecr = str(
                value.get(
                    original_key,
                    ""
                )
            ).upper()

            if related_ecr == active_ecr:

                return {
                    "_benchmark_withheld":
                        True,

                    "reason":
                        (
                            "Record explicitly associated "
                            "with the active benchmark ECR "
                            "was withheld."
                        )
                }


        # --------------------------------------------------
        # DIRECT RELATIONSHIP INVOLVING ACTIVE ECR
        # --------------------------------------------------

        source_id = str(
            value.get(
                "Source_ID",
                ""
            )
        ).upper()

        target_id = str(
            value.get(
                "Target_ID",
                ""
            )
        ).upper()


        if (
            source_id == active_ecr
            or target_id == active_ecr
        ):

            return {
                "_benchmark_withheld":
                    True,

                "reason":
                    (
                        "Direct lifecycle relationship "
                        "involving the active benchmark ECR "
                        "was withheld."
                    )
            }


        # --------------------------------------------------
        # IMPACT ENTITIES
        # --------------------------------------------------

        if (
            is_impact_entity(source_id)
            or is_impact_entity(target_id)
        ):

            return {
                "_benchmark_withheld":
                    True,

                "reason":
                    (
                        "Direct IMPACT-* relationship "
                        "was withheld."
                    )
            }


        entity_like_fields = [
            "entity_id",
            "affected_entity_id",
            "impact_id"
        ]

        for field_name in entity_like_fields:

            if field_name in keys:

                original_key = keys[
                    field_name
                ]

                field_value = str(
                    value.get(
                        original_key,
                        ""
                    )
                )

                if is_impact_entity(
                    field_value
                ):

                    return {
                        "_benchmark_withheld":
                            True,

                        "reason":
                            (
                                "Direct IMPACT-* evidence "
                                "was withheld."
                            )
                    }


        # --------------------------------------------------
        # RECURSIVE SANITIZATION
        # --------------------------------------------------

        sanitized_dict = {}

        for key, item in value.items():

            sanitized_item = (
                sanitize_benchmark_result(
                    item,
                    active_ecr
                )
            )

            if (
                isinstance(
                    sanitized_item,
                    dict
                )
                and sanitized_item.get(
                    "_benchmark_withheld"
                )
            ):
                continue

            sanitized_dict[
                key
            ] = sanitized_item

        return sanitized_dict


    return value


# ==========================================================
# TOOL EXECUTION
# ==========================================================

def execute_tool(
    tool_name,
    arguments,
    benchmark_mode=False,
    active_ecr=None
):

    if tool_name == "get_change":

        return plm_tools.get_change(
            arguments["ecr_id"]
        )


    if tool_name == "get_component":

        return plm_tools.get_component(
            arguments["component_id"]
        )


    if tool_name == "get_related_components":

        return (
            plm_tools.get_related_components(
                arguments["component_id"]
            )
        )


    if tool_name == "get_design_specs":

        return plm_tools.get_design_specs(
            arguments["component_id"]
        )


    if tool_name == "get_requirement":

        return plm_tools.get_requirement(
            arguments["requirement_id"]
        )


    if tool_name == "search_requirements":

        return plm_tools.search_requirements(
            arguments["keyword"]
        )


    if tool_name == "get_bom_for_component":

        return (
            plm_tools.get_bom_for_component(
                arguments["component_id"]
            )
        )


    if tool_name == "get_supplier_for_component":

        component_id = str(
            arguments["component_id"]
        )

        df = plm_tools.plm.suppliers

        result = df[
            df[
                "Primary_Component_ID"
            ].astype(str)
            == component_id
        ]

        return result.to_dict(
            orient="records"
        )


    if tool_name == "get_manufacturing_for_component":

        component_id = str(
            arguments["component_id"]
        )

        df = plm_tools.plm.manufacturing

        result = df[
            df[
                "Component_ID"
            ].astype(str)
            == component_id
        ]

        return result.to_dict(
            orient="records"
        )


    if tool_name == "get_validation_test":

        return plm_tools.get_validation_test(
            arguments["test_id"]
        )


    if tool_name == "get_validation_for_requirement":

        requirement_id = str(
            arguments["requirement_id"]
        )

        df = plm_tools.plm.validation

        result = df[
            df[
                "Requirement_ID"
            ].astype(str)
            == requirement_id
        ]

        return result.to_dict(
            orient="records"
        )


    if tool_name == "get_service_for_component":

        component_id = str(
            arguments["component_id"]
        )

        df = plm_tools.plm.service

        result = df[
            df[
                "Component_ID"
            ].astype(str)
            == component_id
        ]

        records = result.to_dict(
            orient="records"
        )

        if benchmark_mode:

            records = (
                filter_active_ecr_records(
                    records,
                    active_ecr
                )
            )

        return records


    if tool_name == "get_field_failures_for_component":

        component_id = str(
            arguments["component_id"]
        )

        result = field_failures_df[
            field_failures_df[
                "Component_ID"
            ].astype(str)
            == component_id
        ]

        records = result.to_dict(
            orient="records"
        )

        if benchmark_mode:

            records = (
                filter_active_ecr_records(
                    records,
                    active_ecr
                )
            )

        return records


    if tool_name == "get_documents":

        object_id = str(
            arguments["object_id"]
        )

        if is_impact_entity(
            object_id
        ):

            return {
                "blocked":
                    True,

                "reason":
                    "Direct IMPACT-* access is prohibited."
            }

        return plm_tools.get_documents(
            object_id
        )


    if tool_name == "trace_entity":

        entity_id = str(
            arguments["entity_id"]
        )

        if is_impact_entity(
            entity_id
        ):

            return {
                "blocked":
                    True,

                "reason":
                    "Tracing IMPACT-* entities is prohibited."
            }

        trace_result = (
            plm_tools.trace_entity(
                entity_id
            )
        )

        trace_result = (
            remove_impact_relationships(
                trace_result
            )
        )

        if benchmark_mode:

            trace_result = (
                filter_trace_for_benchmark(
                    trace_result,
                    active_ecr
                )
            )

        return trace_result


    if tool_name == "get_entity_details":

        entity_id = str(
            arguments["entity_id"]
        )

        if is_impact_entity(
            entity_id
        ):

            return {
                "blocked":
                    True,

                "reason":
                    "Direct IMPACT-* entity lookup is prohibited."
            }

        return (
            plm_tools.get_entity_details(
                entity_id
            )
        )


    if tool_name == "retrieve_historical_evidence":

        top_k = int(
            arguments["top_k"]
        )

        top_k = max(
            1,
            min(
                top_k,
                10
            )
        )

        exclude_ecr = (
            active_ecr
            if benchmark_mode
            else None
        )

        return retriever.retrieve_evidence(
            query=arguments["query"],
            top_k=top_k,
            exclude_ecr=exclude_ecr
        )


    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# ==========================================================
# TOKEN-USAGE HELPER
# ==========================================================

def update_usage_metrics(
    metrics,
    response
):

    usage = getattr(
        response,
        "usage",
        None
    )

    if usage is None:
        return

    input_tokens = getattr(
        usage,
        "input_tokens",
        0
    ) or 0

    output_tokens = getattr(
        usage,
        "output_tokens",
        0
    ) or 0

    total_tokens = getattr(
        usage,
        "total_tokens",
        0
    ) or 0


    metrics[
        "input_tokens"
    ] += input_tokens

    metrics[
        "output_tokens"
    ] += output_tokens

    metrics[
        "total_tokens"
    ] += total_tokens


# ==========================================================
# TOOL CACHE KEY
# ==========================================================

def build_tool_cache_key(
    tool_name,
    arguments
):

    return (
        tool_name,
        json.dumps(
            arguments,
            sort_keys=True,
            ensure_ascii=False
        )
    )


# ==========================================================
# BENCHMARK WITHHOLDING DETECTION
# ==========================================================

def result_contains_benchmark_withholding(
    value
):

    if isinstance(value, dict):

        if value.get(
            "_benchmark_withheld"
        ):
            return True

        return any(
            result_contains_benchmark_withholding(
                item
            )
            for item in value.values()
        )


    if isinstance(value, list):

        return any(
            result_contains_benchmark_withholding(
                item
            )
            for item in value
        )


    return False


# ==========================================================
# STRUCTURED PREDICTION EXTRACTION
# ==========================================================

def extract_prediction_json(
    response_text
):

    if not response_text:
        return None


    # First try fenced JSON after PREDICTION_JSON.
    pattern = (
        r"PREDICTION_JSON\s*"
        r"```json\s*"
        r"(\{.*?\})"
        r"\s*```"
    )


    match = re.search(
        pattern,
        response_text,
        flags=re.DOTALL
    )


    if match:

        try:
            return json.loads(
                match.group(1)
            )

        except json.JSONDecodeError:
            pass


    # Fallback: allow raw JSON after PREDICTION_JSON.
    marker = "PREDICTION_JSON"

    marker_index = response_text.rfind(
        marker
    )

    if marker_index == -1:
        return None


    remaining = response_text[
        marker_index + len(marker):
    ].strip()


    first_brace = remaining.find(
        "{"
    )

    if first_brace == -1:
        return None


    decoder = json.JSONDecoder()

    try:

        prediction, _ = decoder.raw_decode(
            remaining[
                first_brace:
            ]
        )

        return prediction

    except json.JSONDecodeError:

        return None


# ==========================================================
# PREDICTION VALIDATION
# ==========================================================

def validate_prediction(
    prediction,
    expected_ecr_id
):

    errors = []


    if prediction is None:

        errors.append(
            (
                "Prediction JSON was not found "
                "or could not be parsed."
            )
        )

        return errors


    if not isinstance(
        prediction,
        dict
    ):

        errors.append(
            "Prediction must be a JSON object."
        )

        return errors


    prediction_ecr = str(
        prediction.get(
            "ecr_id",
            ""
        )
    ).upper()


    if prediction_ecr != str(
        expected_ecr_id
    ).upper():

        errors.append(
            (
                "Prediction ECR ID does not match "
                f"{expected_ecr_id}."
            )
        )


    impacts = prediction.get(
        "predicted_impacts"
    )


    if not isinstance(
        impacts,
        list
    ):

        errors.append(
            "predicted_impacts must be a list."
        )

        return errors


    allowed_levels = {
        "Direct",
        "First-order",
        "Second-order",
        "Lifecycle"
    }


    allowed_confidence = {
        "Confirmed",
        "Strong candidate",
        "Review candidate"
    }


    seen_entities = set()


    for index, impact in enumerate(
        impacts,
        start=1
    ):

        if not isinstance(
            impact,
            dict
        ):

            errors.append(
                (
                    f"Prediction item {index} "
                    "is not an object."
                )
            )

            continue


        entity_id = str(
            impact.get(
                "entity_id",
                ""
            )
        ).strip()


        impact_level = impact.get(
            "impact_level"
        )


        confidence = impact.get(
            "confidence"
        )


        reason = impact.get(
            "reason"
        )


        if not entity_id:

            errors.append(
                (
                    f"Prediction item {index} "
                    "has no entity_id."
                )
            )


        if is_impact_entity(
            entity_id
        ):

            errors.append(
                (
                    f"{entity_id} is prohibited "
                    "in predictions."
                )
            )


        if entity_id in seen_entities:

            errors.append(
                (
                    "Duplicate predicted entity: "
                    f"{entity_id}"
                )
            )


        seen_entities.add(
            entity_id
        )


        if impact_level not in allowed_levels:

            errors.append(
                (
                    f"{entity_id} has invalid "
                    f"impact_level: {impact_level}"
                )
            )


        if confidence not in allowed_confidence:

            errors.append(
                (
                    f"{entity_id} has invalid "
                    f"confidence: {confidence}"
                )
            )


        if not isinstance(
            reason,
            str
        ) or not reason.strip():

            errors.append(
                (
                    f"{entity_id} must include "
                    "a non-empty reason."
                )
            )


    return errors


# ==========================================================
# SAVE PREDICTION
# ==========================================================

def save_prediction(
    prediction,
    ecr_id,
    benchmark_mode,
    metrics
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    mode_name = (
        "benchmark"
        if benchmark_mode
        else "normal"
    )


    filename = (
        f"{ecr_id}_"
        f"{mode_name}_"
        f"{timestamp}.json"
    )


    output_path = (
        PREDICTIONS_FOLDER
        / filename
    )


    payload = {

        "system":
            "Aureon Autonomous AI-PLM Agent",

        "model":
            MODEL,

        "ecr_id":
            ecr_id,

        "benchmark_mode":
            benchmark_mode,

        "created_at":
            datetime.now().isoformat(),

        "metrics":
            metrics,

        "prediction":
            prediction
    }


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False
        )


    return output_path


# ==========================================================
# AUTONOMOUS AGENT LOOP
# ==========================================================

def analyse_change_agentically(
    ecr_id,
    benchmark_mode=False,
    max_iterations=MAX_ITERATIONS,
    max_tool_calls=MAX_TOOL_CALLS
):

    start_time = time.perf_counter()


    ecr_id = (
        ecr_id
        .strip()
        .upper()
    )


    mode_text = (
        "BENCHMARK-SAFE"
        if benchmark_mode
        else "NORMAL ENGINEERING"
    )


    initial_prompt = f"""
Perform an autonomous PLM engineering change-impact
assessment for:

Engineering change: {ecr_id}

Mode: {mode_text}

Start by retrieving the engineering change record.

Then inspect the active ECR lifecycle trace unless it is
filtered by benchmark mode.

Determine which additional evidence is genuinely necessary.

Do not call every tool.

Prioritize evidence that can materially change impact
identification, engineering confidence, risk assessment or
recommended action.

Aim to finish using approximately 12 to 25 useful tool calls.

A hard maximum of {max_tool_calls} tool requests is enforced.

Never request an IMPACT-* entity.

Do not use evaluator ground truth.

Do not use get_change_impacts or analyze_change.

In benchmark mode, do not attempt to reconstruct evidence
that the application has withheld.

When sufficient evidence has been gathered, stop using tools
and produce the final engineering assessment.

The final response must include the required PREDICTION_JSON
block.
"""


    input_items = [
        {
            "role": "user",
            "content": initial_prompt
        }
    ]


    tool_log = []

    tool_cache = {}


    metrics = {

        "reasoning_cycles":
            0,

        "tool_calls_requested":
            0,

        "tool_calls_executed":
            0,

        "blocked_calls":
            0,

        "benchmark_withheld_results":
            0,

        "cache_hits":
            0,

        "tool_errors":
            0,

        "input_tokens":
            0,

        "output_tokens":
            0,

        "total_tokens":
            0,

        "elapsed_seconds":
            0.0
    }


    # ======================================================
    # REASONING LOOP
    # ======================================================

    for iteration in range(
        1,
        max_iterations + 1
    ):

        metrics[
            "reasoning_cycles"
        ] = iteration


        print(
            f"\n--- Agent reasoning cycle "
            f"{iteration} ---"
        )


        response = client.responses.create(
            model=MODEL,
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS,
            input=input_items
        )


        update_usage_metrics(
            metrics,
            response
        )


        input_items.extend(
            response.output
        )


        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]


        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        if not function_calls:

            metrics[
                "elapsed_seconds"
            ] = round(
                time.perf_counter()
                - start_time,
                3
            )

            return {

                "answer":
                    response.output_text,

                "tool_log":
                    tool_log,

                "iterations":
                    iteration,

                "metrics":
                    metrics
            }


        # ==================================================
        # PROCESS TOOL CALLS
        # ==================================================

        for function_call in function_calls:

            metrics[
                "tool_calls_requested"
            ] += 1


            tool_name = (
                function_call.name
            )


            try:

                arguments = json.loads(
                    function_call.arguments
                )

            except json.JSONDecodeError:

                arguments = {}


            print(
                f"Tool selected: "
                f"{tool_name}"
            )

            print(
                f"Arguments: "
                f"{arguments}"
            )


            cache_key = (
                build_tool_cache_key(
                    tool_name,
                    arguments
                )
            )


            # ==============================================
            # DUPLICATE CALL CACHE
            # ==============================================

            if cache_key in tool_cache:

                result = tool_cache[
                    cache_key
                ]

                metrics[
                    "cache_hits"
                ] += 1


                tool_log.append(
                    {
                        "cycle":
                            iteration,

                        "tool":
                            tool_name,

                        "arguments":
                            arguments,

                        "status":
                            "cache_hit"
                    }
                )


                print(
                    "Duplicate request detected - "
                    "cached result returned."
                )


            # ==============================================
            # TOOL BUDGET
            # ==============================================

            elif (
                metrics[
                    "tool_calls_executed"
                ]
                >= max_tool_calls
            ):

                result = {

                    "blocked":
                        True,

                    "reason":
                        (
                            "Tool-call budget exhausted. "
                            "Use evidence already collected "
                            "and produce the final assessment."
                        )
                }


                metrics[
                    "blocked_calls"
                ] += 1


                tool_log.append(
                    {
                        "cycle":
                            iteration,

                        "tool":
                            tool_name,

                        "arguments":
                            arguments,

                        "status":
                            "budget_blocked"
                    }
                )


                print(
                    "Tool budget exhausted - "
                    "request blocked."
                )


            # ==============================================
            # EXECUTE TOOL
            # ==============================================

            else:

                try:

                    result = execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        benchmark_mode=benchmark_mode,
                        active_ecr=ecr_id
                    )


                    result = clean_json_value(
                        result
                    )


                    # --------------------------------------
                    # UNIVERSAL BENCHMARK FIREWALL
                    # --------------------------------------

                    if benchmark_mode:

                        result = (
                            sanitize_benchmark_result(
                                result,
                                active_ecr=ecr_id
                            )
                        )


                    # --------------------------------------
                    # STATUS
                    # --------------------------------------

                    if (
                        isinstance(result, dict)
                        and result.get(
                            "blocked"
                        )
                    ):

                        metrics[
                            "blocked_calls"
                        ] += 1

                        status = (
                            "security_blocked"
                        )


                    elif (
                        isinstance(result, dict)
                        and result.get(
                            "_benchmark_withheld"
                        )
                    ):

                        metrics[
                            "benchmark_withheld_results"
                        ] += 1

                        status = (
                            "benchmark_withheld"
                        )


                    elif (
                        result_contains_benchmark_withholding(
                            result
                        )
                    ):

                        metrics[
                            "benchmark_withheld_results"
                        ] += 1

                        metrics[
                            "tool_calls_executed"
                        ] += 1

                        status = (
                            "success_filtered"
                        )


                    else:

                        metrics[
                            "tool_calls_executed"
                        ] += 1

                        status = (
                            "success"
                        )


                    tool_cache[
                        cache_key
                    ] = result


                    tool_log.append(
                        {
                            "cycle":
                                iteration,

                            "tool":
                                tool_name,

                            "arguments":
                                arguments,

                            "status":
                                status
                        }
                    )


                except Exception as error:

                    result = {
                        "error":
                            str(error)
                    }


                    metrics[
                        "tool_errors"
                    ] += 1


                    tool_log.append(
                        {
                            "cycle":
                                iteration,

                            "tool":
                                tool_name,

                            "arguments":
                                arguments,

                            "status":
                                "error",

                            "error":
                                str(error)
                        }
                    )


            print(
                "Tool completed."
            )


            # ==============================================
            # RETURN TOOL OUTPUT TO MODEL
            # ==============================================

            input_items.append(
                {
                    "type":
                        "function_call_output",

                    "call_id":
                        function_call.call_id,

                    "output":
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str
                        )
                }
            )


    # ======================================================
    # ITERATION SAFETY STOP
    # ======================================================

    metrics[
        "elapsed_seconds"
    ] = round(
        time.perf_counter()
        - start_time,
        3
    )


    return {

        "answer": (
            "Agent stopped after reaching the maximum "
            f"of {max_iterations} reasoning cycles. "
            "Review the tool trace and evidence budget "
            "before increasing the limit."
        ),

        "tool_log":
            tool_log,

        "iterations":
            max_iterations,

        "metrics":
            metrics
    }


# ==========================================================
# TOOL TRACE
# ==========================================================

def print_tool_trace(
    tool_log
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "AGENT TOOL TRACE"
    )

    print(
        "=" * 90
    )


    if not tool_log:

        print(
            "No tools were called."
        )

        return


    for number, entry in enumerate(
        tool_log,
        start=1
    ):

        print(
            f"\n{number}. "
            f"Cycle {entry['cycle']} - "
            f"{entry['tool']}"
        )

        print(
            f"   Arguments: "
            f"{entry['arguments']}"
        )

        print(
            f"   Status: "
            f"{entry['status']}"
        )


# ==========================================================
# METRICS
# ==========================================================

def print_metrics(
    metrics
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "AGENT EXECUTION METRICS"
    )

    print(
        "=" * 90
    )


    print(
        f"Reasoning cycles: "
        f"{metrics['reasoning_cycles']}"
    )

    print(
        f"Tool requests: "
        f"{metrics['tool_calls_requested']}"
    )

    print(
        f"Tools executed: "
        f"{metrics['tool_calls_executed']}"
    )

    print(
        f"Duplicate/cache hits: "
        f"{metrics['cache_hits']}"
    )

    print(
        f"Blocked calls: "
        f"{metrics['blocked_calls']}"
    )

    print(
        f"Benchmark-withheld results: "
        f"{metrics['benchmark_withheld_results']}"
    )

    print(
        f"Tool errors: "
        f"{metrics['tool_errors']}"
    )

    print(
        f"Input tokens: "
        f"{metrics['input_tokens']}"
    )

    print(
        f"Output tokens: "
        f"{metrics['output_tokens']}"
    )

    print(
        f"Total tokens: "
        f"{metrics['total_tokens']}"
    )

    print(
        f"Elapsed time: "
        f"{metrics['elapsed_seconds']} seconds"
    )


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 90
    )

    print(
        "AUREON AUTONOMOUS AI-PLM AGENT"
    )

    print(
        "Phase 5C.0 - Structured Prediction Output"
    )

    print(
        "=" * 90
    )


    print(
        f"\nModel: {MODEL}"
    )

    print(
        f"Maximum tool executions: "
        f"{MAX_TOOL_CALLS}"
    )


    ecr_id = input(
        "\nEnter ECR ID "
        "(example ECR-001): "
    ).strip().upper()


    print(
        "\nAnalysis mode:"
    )

    print(
        "1 - Normal autonomous analysis"
    )

    print(
        "2 - Benchmark-safe autonomous analysis"
    )


    mode = input(
        "\nSelect mode [1]: "
    ).strip()


    benchmark_mode = (
        mode == "2"
    )


    if benchmark_mode:

        print(
            "\nBenchmark-safe mode enabled."
        )

        print(
            "Same-ECR historical/service/field "
            "evidence will be filtered."
        )

        print(
            "Direct active-ECR lifecycle "
            "relationships will be hidden."
        )

        print(
            "All tool outputs pass through the "
            "universal benchmark sanitizer."
        )


    else:

        print(
            "\nNormal autonomous mode enabled."
        )


    print(
        "\nThe agent will autonomously select PLM tools."
    )

    print(
        "Direct IMPACT-* access is prohibited."
    )

    print(
        "Duplicate calls are cached."
    )


    result = (
        analyse_change_agentically(
            ecr_id=ecr_id,
            benchmark_mode=benchmark_mode
        )
    )


    # ======================================================
    # TOOL TRACE
    # ======================================================

    print_tool_trace(
        result[
            "tool_log"
        ]
    )


    # ======================================================
    # EXECUTION METRICS
    # ======================================================

    print_metrics(
        result[
            "metrics"
        ]
    )


    # ======================================================
    # FINAL RESPONSE
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL AUTONOMOUS AGENT RESPONSE"
    )

    print(
        "=" * 90
    )


    print(
        result[
            "answer"
        ]
    )


    # ======================================================
    # STRUCTURED PREDICTION PROCESSING
    # ======================================================

    prediction = extract_prediction_json(
        result[
            "answer"
        ]
    )


    prediction_errors = (
        validate_prediction(
            prediction,
            expected_ecr_id=ecr_id
        )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "STRUCTURED PREDICTION STATUS"
    )

    print(
        "=" * 90
    )


    if prediction_errors:

        print(
            "\nPrediction validation FAILED."
        )

        for error in prediction_errors:

            print(
                f"- {error}"
            )


    else:

        print(
            "\nPrediction validation PASSED."
        )


        predicted_entities = [
            impact[
                "entity_id"
            ]
            for impact in prediction[
                "predicted_impacts"
            ]
        ]


        print(
            f"\nPredicted entities: "
            f"{len(predicted_entities)}"
        )


        for entity_id in (
            predicted_entities
        ):

            print(
                f"  - {entity_id}"
            )


        prediction_file = (
            save_prediction(
                prediction=prediction,
                ecr_id=ecr_id,
                benchmark_mode=benchmark_mode,
                metrics=result[
                    "metrics"
                ]
            )
        )


        print(
            "\nPrediction saved to:"
        )

        print(
            prediction_file
        )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "AUTONOMOUS ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )