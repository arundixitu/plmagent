import os
import sys
from pathlib import Path


# ==========================================================
# IMPORTANT WINDOWS DLL INITIALIZATION
# ==========================================================

# On this Windows environment, PyTorch must be loaded
# before pandas / chromadb / other native libraries.
import torch


from dotenv import load_dotenv
from openai import OpenAI


# ==========================================================
# PROJECT PATH SETUP
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# LOCAL PROJECT IMPORTS
# ==========================================================

# Keep RAG import before PLMTools because of the Windows
# native-library loading behaviour observed on this machine.
from rag.retriever import AureonRetriever
from tools.plm_tools import PLMTools
from agent.evidence_collector import collect_structured_evidence


# ==========================================================
# LOAD ENVIRONMENT
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


# ==========================================================
# OPENAI CLIENT
# ==========================================================

client = OpenAI(
    api_key=API_KEY
)


# ==========================================================
# INITIALIZE PLM CAPABILITIES
# ==========================================================

plm_tools = PLMTools()

retriever = AureonRetriever()


# ==========================================================
# AGENT INSTRUCTIONS
# ==========================================================

AGENT_INSTRUCTIONS = """
You are the Aureon PLM Engineering Agent.

You support engineering change-impact analysis for the
Aureon E-Move X1 product lifecycle.

You are not a generic chatbot.

You are an evidence-grounded PLM engineering reasoning
assistant.

Your task is to analyse engineering changes using:

1. deterministic structured PLM evidence;
2. explicit lifecycle relationships;
3. design specifications;
4. requirements;
5. validation information;
6. BOM/configuration information;
7. supplier information;
8. manufacturing information;
9. service information;
10. field-failure evidence;
11. controlled-document information; and
12. historical semantic evidence retrieved from the
    engineering RAG system.


============================================================
EVIDENCE RULES
============================================================

Your reasoning must follow this sequence:

CLAIM
→ EVIDENCE
→ ENGINEERING REASONING
→ CONSEQUENCE
→ RISK
→ RECOMMENDED ACTION


Always distinguish between:

- Known from PLM data
- Explicit relationship
- Historical evidence
- Calculated value
- Engineering inference
- Unknown or missing information


Never present engineering inference as confirmed PLM fact.


============================================================
DO NOT INVENT
============================================================

Never invent:

- dimensions
- mass values
- voltage values
- current values
- costs
- suppliers
- manufacturing processes
- test results
- validation status
- regulatory conclusions
- safety conclusions
- component specifications
- requirement values
- service history
- field failures


If information is absent, state that it is unknown or
requires verification.


============================================================
RELATIONSHIP INTERPRETATION RULE
============================================================

Relationship IDs from different PLM tables are local to
their respective tables.

For example:

05_Product_Relationships.xlsx

and

18_Relationship_Matrix.xlsx

may both contain a Relationship_ID such as REL-007.

This does NOT automatically indicate a data conflict.

Always identify the source relationship table before
interpreting the relationship.

Use:

PRODUCT RELATIONSHIP

for relationships originating from
05_Product_Relationships.xlsx.

Use:

LIFECYCLE TRACE RELATIONSHIP

for relationships originating from
18_Relationship_Matrix.xlsx.


============================================================
IMPACT REASONING RULES
============================================================

A component being connected to another component does NOT
automatically mean that the connected component requires a
design change.

Use wording such as:

- review required
- compatibility should be verified
- potential impact
- engineering analysis required
- validation should be considered

unless the PLM evidence explicitly establishes the impact.


Analyse impacts at multiple levels where evidence supports
them:

DIRECT IMPACT

FIRST-ORDER IMPACT

SECOND-ORDER IMPACT

LIFECYCLE IMPACT


Consider the following impact dimensions:

- requirements
- product architecture
- mechanical design
- electrical systems
- thermal systems
- software/control
- BOM/configuration
- suppliers
- manufacturing
- validation
- service
- field performance
- documentation
- safety
- customer/product performance


============================================================
HISTORICAL EVIDENCE RULE
============================================================

Historical records provide engineering context.

They do NOT prove that:

- the present ECR has the same root cause;
- the historical solution is appropriate;
- the proposed ECR will resolve a historical problem; or
- the same components must change.

Use historical evidence to support investigation and
comparison, not as automatic proof.


============================================================
CALCULATION RULE
============================================================

Simple engineering arithmetic may be performed if all input
values are explicitly available.

When performing a calculation:

1. state the input values;
2. state the assumption;
3. state the calculation;
4. identify the result as calculated rather than stored PLM
   evidence.

Do not silently assume missing engineering values.


============================================================
VALIDATION RULE
============================================================

Validation of an existing baseline configuration does NOT
automatically validate a proposed changed configuration.

If an affected specification or requirement changes,
identify the need for review or re-validation where
appropriate.


============================================================
BENCHMARK RULE
============================================================

During benchmark mode, evidence directly belonging to the
ECR under evaluation may be deliberately withheld.

Do not infer that absence means the evidence never existed.

Never request or attempt to access:

99_Ground_Truth_Impact_Matrix.xlsx

The ground-truth dataset is evaluator-only.


============================================================
OUTPUT FORMAT
============================================================

Return the engineering assessment using exactly these main
sections:

CHANGE SUMMARY

CONFIRMED PLM EVIDENCE

DIRECT IMPACT

LIKELY FIRST-ORDER / SECOND-ORDER IMPACTS

REQUIREMENTS IMPACT

DESIGN / COMPONENT IMPACT

BOM / CONFIGURATION IMPACT

VALIDATION IMPACT

MANUFACTURING / SUPPLIER IMPACT

SERVICE / FIELD IMPACT

DOCUMENTATION IMPACT

HISTORICAL EVIDENCE

UNKNOWN OR MISSING INFORMATION

RECOMMENDED ENGINEERING ACTIONS

PLM LEARNING TAKEAWAY


Use clear engineering language.

Explicitly distinguish confirmed evidence from inferred
impact.

Do not exaggerate certainty.
"""


# ==========================================================
# BUILD STRUCTURED EVIDENCE
# ==========================================================

def build_structured_evidence(
    ecr_id,
    benchmark_mode=False
):
    """
    Build enriched deterministic PLM evidence.

    Data gathering is handled by evidence_collector.py
    so that the reasoning agent remains separated from
    data-access logic.
    """

    return collect_structured_evidence(
        tools=plm_tools,
        ecr_id=ecr_id,
        benchmark_mode=benchmark_mode
    )


# ==========================================================
# BUILD SEMANTIC / RAG EVIDENCE
# ==========================================================

def build_rag_evidence(
    change,
    exclude_ecr=None,
    top_k=10
):
    """
    Retrieve historical semantic evidence.

    exclude_ecr is used during controlled benchmarking to
    prevent retrieval of historical evidence directly
    belonging to the ECR under evaluation.
    """

    query = (
        f"Historical engineering evidence relevant to "
        f"{change['Change_Title']}. "
        f"Primary component: "
        f"{change['Primary_Component_ID']}. "
        f"Change description: "
        f"{change['Change_Description']}. "
        f"Identify relevant component, thermal, electrical, "
        f"mechanical, validation, manufacturing, supplier, "
        f"service and field evidence."
    )

    return retriever.retrieve_evidence(
        query=query,
        top_k=top_k,
        exclude_ecr=exclude_ecr
    )


# ==========================================================
# GENERAL RECORD FORMATTER
# ==========================================================

def format_records(records):
    """
    Convert a list of PLM dictionaries into readable text.
    """

    if not records:
        return "No records found."

    return "\n".join(
        str(record)
        for record in records
    )


# ==========================================================
# FORMAT STRUCTURED EVIDENCE
# ==========================================================

def format_structured_evidence(
    evidence
):
    """
    Convert the deterministic evidence package into a
    structured text representation for the reasoning model.
    """

    sections = []

    # ------------------------------------------------------
    # ENGINEERING CHANGE
    # ------------------------------------------------------

    sections.append(
        "ENGINEERING CHANGE\n"
        "------------------\n"
        + str(
            evidence["change"]
        )
    )

    # ------------------------------------------------------
    # PRIMARY COMPONENT
    # ------------------------------------------------------

    sections.append(
        "PRIMARY COMPONENT\n"
        "-----------------\n"
        + str(
            evidence[
                "primary_component"
            ]
        )
    )

    # ------------------------------------------------------
    # PRODUCT RELATIONSHIPS
    # ------------------------------------------------------

    sections.append(
        "PRODUCT RELATIONSHIPS\n"
        "Source: 05_Product_Relationships.xlsx\n"
        "-------------------------------------\n"
        + format_records(
            evidence[
                "product_relationships"
            ]
        )
    )

    # ------------------------------------------------------
    # RELATED COMPONENT DETAILS
    # ------------------------------------------------------

    sections.append(
        "RELATED COMPONENT DETAILS\n"
        "-------------------------\n"
        + format_records(
            evidence[
                "related_components"
            ]
        )
    )

    # ------------------------------------------------------
    # RELATED COMPONENT SUPPORTING EVIDENCE
    # ------------------------------------------------------

    sections.append(
        "RELATED COMPONENT SUPPORTING EVIDENCE\n"
        "-------------------------------------\n"
        + format_records(
            evidence[
                "related_component_evidence"
            ]
        )
    )

    # ------------------------------------------------------
    # DESIGN SPECIFICATIONS
    # ------------------------------------------------------

    sections.append(
        "DESIGN SPECIFICATIONS\n"
        "---------------------\n"
        + format_records(
            evidence[
                "design_specifications"
            ]
        )
    )

    # ------------------------------------------------------
    # REQUIREMENTS
    # ------------------------------------------------------

    sections.append(
        "LINKED REQUIREMENTS\n"
        "-------------------\n"
        + format_records(
            evidence[
                "requirements"
            ]
        )
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    sections.append(
        "VALIDATION RECORDS\n"
        "------------------\n"
        + format_records(
            evidence[
                "validation_records"
            ]
        )
    )

    # ------------------------------------------------------
    # BOM
    # ------------------------------------------------------

    sections.append(
        "BOM RECORDS\n"
        "-----------\n"
        + format_records(
            evidence[
                "bom_records"
            ]
        )
    )

    # ------------------------------------------------------
    # SUPPLIERS
    # ------------------------------------------------------

    sections.append(
        "SUPPLIER RECORDS\n"
        "----------------\n"
        + format_records(
            evidence[
                "supplier_records"
            ]
        )
    )

    # ------------------------------------------------------
    # MANUFACTURING
    # ------------------------------------------------------

    sections.append(
        "MANUFACTURING RECORDS\n"
        "---------------------\n"
        + format_records(
            evidence[
                "manufacturing_records"
            ]
        )
    )

    # ------------------------------------------------------
    # SERVICE
    # ------------------------------------------------------

    sections.append(
        "SERVICE RECORDS\n"
        "---------------\n"
        + format_records(
            evidence[
                "service_records"
            ]
        )
    )

    # ------------------------------------------------------
    # FIELD FAILURES
    # ------------------------------------------------------

    sections.append(
        "FIELD FAILURE RECORDS\n"
        "---------------------\n"
        + format_records(
            evidence[
                "field_failure_records"
            ]
        )
    )

    # ------------------------------------------------------
    # DOCUMENT CONTROL
    # ------------------------------------------------------

    sections.append(
        "CONTROLLED DOCUMENTS\n"
        "--------------------\n"
        + format_records(
            evidence[
                "controlled_documents"
            ]
        )
    )

    # ------------------------------------------------------
    # LIFECYCLE TRACE
    # ------------------------------------------------------

    lifecycle_trace = evidence[
        "lifecycle_trace"
    ]

    sections.append(
        "FIRST-HOP LIFECYCLE TRACE\n"
        "Source: 18_Relationship_Matrix.xlsx\n"
        "-----------------------------------\n"
        + format_records(
            lifecycle_trace[
                "first_hop_relationships"
            ]
        )
    )

    sections.append(
        "SECOND-HOP LIFECYCLE TRACE\n"
        "Source: 18_Relationship_Matrix.xlsx\n"
        "-----------------------------------\n"
        + format_records(
            lifecycle_trace[
                "second_hop_relationships"
            ]
        )
    )

    return "\n\n".join(
        sections
    )


# ==========================================================
# FORMAT RAG EVIDENCE
# ==========================================================

def format_rag_evidence(
    results
):
    """
    Format historical semantic retrieval results.
    """

    if not results:
        return "No historical semantic evidence retrieved."

    formatted = []

    for index, result in enumerate(
        results,
        start=1
    ):
        formatted.append(
            f"{index}. "
            f"[Category: {result['category']}] "
            f"[Record: {result['record_id']}] "
            f"[Source: {result['source_file']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(
        formatted
    )


# ==========================================================
# RUN ENGINEERING ANALYSIS
# ==========================================================

def analyse_change(
    ecr_id,
    benchmark_mode=False
):
    """
    Perform an evidence-grounded engineering change-impact
    assessment.
    """

    # ------------------------------------------------------
    # STRUCTURED EVIDENCE
    # ------------------------------------------------------

    structured = build_structured_evidence(
        ecr_id=ecr_id,
        benchmark_mode=benchmark_mode
    )

    if structured is None:
        return (
            f"Engineering change {ecr_id} "
            f"was not found in the Aureon PLM dataset."
        )

    change = structured["change"]

    # ------------------------------------------------------
    # BENCHMARK CONTROL
    # ------------------------------------------------------

    exclude_ecr = (
        ecr_id
        if benchmark_mode
        else None
    )

    # ------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # ------------------------------------------------------

    rag_results = build_rag_evidence(
        change=change,
        exclude_ecr=exclude_ecr,
        top_k=10
    )

    # ------------------------------------------------------
    # FORMAT EVIDENCE
    # ------------------------------------------------------

    structured_text = (
        format_structured_evidence(
            structured
        )
    )

    rag_text = (
        format_rag_evidence(
            rag_results
        )
    )

    # ------------------------------------------------------
    # BUILD REASONING REQUEST
    # ------------------------------------------------------

    user_prompt = f"""
Analyse engineering change {ecr_id}.

============================================================
STRUCTURED PLM EVIDENCE
============================================================

{structured_text}


============================================================
HISTORICAL / SEMANTIC RAG EVIDENCE
============================================================

{rag_text}


============================================================
ANALYSIS MODE
============================================================

Benchmark mode: {benchmark_mode}


============================================================
ENGINEERING TASK
============================================================

Produce an evidence-grounded engineering change-impact
assessment.

Use structured PLM evidence as the primary source of truth.

Use historical RAG evidence only as supporting engineering
context.

Do not assume every connected component must change.

For each significant conclusion, determine whether it is:

1. confirmed PLM evidence;
2. explicit relationship evidence;
3. historical evidence;
4. calculated;
5. engineering inference; or
6. unknown / missing information.

Identify cross-lifecycle consequences where supported by
evidence.

Where evidence is insufficient, recommend review,
verification, analysis or validation rather than asserting
a confirmed design impact.
"""

    # ------------------------------------------------------
    # MODEL REASONING
    # ------------------------------------------------------

    response = client.responses.create(
        model=MODEL,
        instructions=AGENT_INSTRUCTIONS,
        input=user_prompt
    )

    return response.output_text


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

if __name__ == "__main__":

    print(
        "\nAUREON AI-PLM ENGINEERING AGENT V1.1"
    )

    print(
        "=" * 90
    )

    print(
        "\nModel:",
        MODEL
    )

    ecr_id = input(
        "\nEnter ECR ID "
        "(example ECR-001): "
    ).strip().upper()

    print(
        "\nAnalysis mode:"
    )

    print(
        "1 - Normal engineering analysis"
    )

    print(
        "2 - Benchmark-safe analysis"
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
            "Direct historical evidence belonging to "
            f"{ecr_id} will be withheld."
        )

    else:

        print(
            "\nNormal engineering analysis mode enabled."
        )

    print(
        "\nCollecting deterministic PLM evidence..."
    )

    print(
        "Retrieving historical engineering evidence..."
    )

    print(
        "Running AI engineering reasoning..."
    )

    result = analyse_change(
        ecr_id=ecr_id,
        benchmark_mode=benchmark_mode
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "AGENT RESPONSE"
    )

    print(
        "=" * 90
    )

    print(result)

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )