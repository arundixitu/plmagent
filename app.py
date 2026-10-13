# ==========================================================
# AUREON AI-PLM ENGINEERING LAB
# Final Streamlit Frontend
#
# Designed and Developed by:
# Dr. Arun C Dixit
# Associate Professor, ME
# VVCE, Mysuru
# ==========================================================


# ==========================================================
# IMPORTANT WINDOWS INITIALIZATION
# ==========================================================
# PyTorch must be imported before pandas / Chroma /
# sentence-transformers in this Windows environment.

import torch


# ==========================================================
# STANDARD IMPORTS
# ==========================================================

import os
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================================
# VERIFIED AUREON BACKEND IMPORTS
# ==========================================================

from rag.retriever import AureonRetriever
from agent.plm_agent import analyse_change


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Aureon AI-PLM Engineering Lab",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = PROJECT_ROOT / "assets"

LOGO_PATH = ASSETS_DIR / "aureon_logo.png"

SCOOTER_PATH = ASSETS_DIR / "aureon_scooter.png"

ARCHITECTURE_PATH = (
    ASSETS_DIR
    / "aureon_architecture.png"
)

DASHBOARD_PREVIEW_PATH = (
    ASSETS_DIR
    / "aureon_dashboard_preview.png"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Aureon_EMove_X1_PLM_Agent_Dataset_V1_COMPLETE.xlsx"
)


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna",
)


# ==========================================================
# CACHED BACKEND RESOURCES
# ==========================================================

@st.cache_resource
def get_retriever():
    """
    Load Aureon semantic retriever once.
    """
    return AureonRetriever()


@st.cache_resource
def get_openai_client():
    """
    Create OpenAI client only when the Ask Agent
    function actually needs it.
    """

    if not OPENAI_API_KEY:
        return None

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ==========================================================
# LOAD EXCEL WORKBOOK
# ==========================================================

@st.cache_data
def load_plm_workbook(filepath):
    """
    Load all sheets from the combined Aureon PLM workbook.
    """

    excel_file = pd.ExcelFile(
        filepath
    )

    sheets = {}

    for sheet_name in excel_file.sheet_names:

        sheets[sheet_name] = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
        )

    return sheets


# ==========================================================
# STRING / DATA HELPERS
# ==========================================================

def normalize_name(text):

    return (
        str(text)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def find_sheet(
    sheets,
    possible_names,
):
    """
    Locate a sheet even when naming conventions vary.
    """

    normalized_sheets = {
        normalize_name(name): name
        for name in sheets.keys()
    }

    # Exact normalized match
    for candidate in possible_names:

        normalized_candidate = normalize_name(
            candidate
        )

        if (
            normalized_candidate
            in normalized_sheets
        ):

            actual_name = normalized_sheets[
                normalized_candidate
            ]

            return sheets[
                actual_name
            ]

    # Partial normalized match
    for candidate in possible_names:

        normalized_candidate = normalize_name(
            candidate
        )

        for (
            normalized_sheet,
            actual_name
        ) in normalized_sheets.items():

            if (
                normalized_candidate
                in normalized_sheet
            ):

                return sheets[
                    actual_name
                ]

    return pd.DataFrame()


def find_column(
    df,
    possible_names,
):
    """
    Locate a DataFrame column using flexible matching.
    """

    if df.empty:
        return None

    normalized_columns = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in possible_names:

        normalized_candidate = normalize_name(
            candidate
        )

        if (
            normalized_candidate
            in normalized_columns
        ):

            return normalized_columns[
                normalized_candidate
            ]

    return None


def safe_string(value):
    """
    Convert values to clean strings.
    """

    if pd.isna(value):
        return ""

    return str(value)


def dataframe_records(df):
    """
    Convert dataframe to JSON-friendly records.
    """

    if df.empty:
        return []

    clean_df = (
        df.copy()
        .where(
            pd.notnull(df),
            None,
        )
    )

    return clean_df.to_dict(
        orient="records"
    )


# ==========================================================
# VERIFY DATA FILE
# ==========================================================

if not DATA_FILE.exists():

    st.error(
        f"""
        The Aureon PLM workbook could not be found.

        Expected location:

        {DATA_FILE}
        """
    )

    st.stop()


try:

    PLM_DATA = load_plm_workbook(
        DATA_FILE
    )

except Exception as error:

    st.error(
        "The Aureon PLM workbook could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ==========================================================
# IDENTIFY WORKBOOK TABLES
# ==========================================================

components_df = find_sheet(
    PLM_DATA,
    [
        "Product Components",
        "Components",
        "Component Master",
        "Product_Component_Master",
        "BOM",
    ],
)


requirements_df = find_sheet(
    PLM_DATA,
    [
        "Product Requirements",
        "Requirements",
        "Product_Requirements",
    ],
)


bom_df = find_sheet(
    PLM_DATA,
    [
        "BOM",
        "Bill of Materials",
        "Product BOM",
    ],
)


changes_df = find_sheet(
    PLM_DATA,
    [
        "Engineering Change Log",
        "Engineering Changes",
        "Engineering_Change_Log",
    ],
)


design_df = find_sheet(
    PLM_DATA,
    [
        "Design Specifications",
        "Design_Specifications",
        "Specifications",
    ],
)


manufacturing_df = find_sheet(
    PLM_DATA,
    [
        "Manufacturing Routing",
        "Manufacturing",
        "Manufacturing_Routing",
    ],
)


validation_df = find_sheet(
    PLM_DATA,
    [
        "Validation Tests",
        "Validation",
        "Validation_Tests",
    ],
)


service_df = find_sheet(
    PLM_DATA,
    [
        "Service Records",
        "Service",
        "Service_Records",
    ],
)


supplier_df = find_sheet(
    PLM_DATA,
    [
        "Supplier Records",
        "Suppliers",
        "Supplier_Records",
    ],
)


relationships_df = find_sheet(
    PLM_DATA,
    [
        "Relationships",
        "Product Relationships",
        "Relationship Matrix",
    ],
)


impact_df = find_sheet(
    PLM_DATA,
    [
        "Change Impact Links",
        "Change_Impact_Links",
        "Impact Links",
    ],
)


field_df = find_sheet(
    PLM_DATA,
    [
        "Field Failure",
        "Field Failures",
        "Field_Failure_Records",
        "Field Failure Records",
    ],
)


documents_df = find_sheet(
    PLM_DATA,
    [
        "Controlled Documents",
        "Documents",
        "Document Control",
    ],
)


# ==========================================================
# GENERAL UI HELPERS
# ==========================================================

def show_image_if_exists(
    image_path,
    width=None,
    use_container_width=False,
):

    if image_path.exists():

        st.image(
            str(image_path),
            width=width,
            use_container_width=use_container_width,
        )

    else:

        st.warning(
            f"Image not found: {image_path.name}"
        )


def metric_card(
    label,
    value,
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(
    title,
    body,
):

    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-title">
                {title}
            </div>
            <div class="feature-body">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def searchable_dataframe(
    df,
    search_key,
):

    if df.empty:

        st.warning(
            "This dataset could not be located in the workbook."
        )

        return

    search_text = st.text_input(
        "Search this table",
        key=search_key,
        placeholder=(
            "Enter ID, component, requirement, "
            "status or keyword..."
        ),
    )

    display_df = df.copy()

    if search_text:

        mask = (
            display_df
            .astype(str)
            .apply(
                lambda row:
                row.str.contains(
                    search_text,
                    case=False,
                    na=False,
                ).any(),
                axis=1,
            )
        )

        display_df = display_df[
            mask
        ]

    st.caption(
        f"Showing {len(display_df)} "
        f"of {len(df)} records"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def related_rows(
    df,
    entity_id,
):

    if df.empty:
        return pd.DataFrame()

    mask = (
        df.astype(str)
        .apply(
            lambda row:
            row.str.contains(
                str(entity_id),
                case=False,
                na=False,
                regex=False,
            ).any(),
            axis=1,
        )
    )

    return df[
        mask
    ]


# ==========================================================
# ENTITY IDENTIFICATION
# ==========================================================

ENTITY_PATTERN = (
    r"\b(?:"
    r"PROD|BAT|BMS|ENC|MTR|INV|CHR|CHG|CAB|CON|"
    r"THM|FAN|FRM|SW|SNS|TCU|"
    r"REQ|TEST|MFG|SUP|SRV|FIELD|DOC|ECR|"
    r"BOM|SPEC|IMPACT"
    r")-[A-Z0-9-]+\b"
)


def extract_entity_ids(text):

    matches = re.findall(
        ENTITY_PATTERN,
        str(text).upper(),
    )

    return list(
        dict.fromkeys(
            matches
        )
    )


# ==========================================================
# ASK PLM AGENT — STRUCTURED EVIDENCE RETRIEVAL
# ==========================================================

def search_dataframe_for_text(
    df,
    text,
):
    """
    Search every column of a PLM dataframe.
    """

    if df.empty:
        return pd.DataFrame()

    mask = (
        df.astype(str)
        .apply(
            lambda row:
            row.str.contains(
                str(text),
                case=False,
                na=False,
                regex=False,
            ).any(),
            axis=1,
        )
    )

    return df[
        mask
    ]


def collect_entity_evidence(
    entity_id,
):
    """
    Retrieve matching records for an entity from
    every visible PLM lifecycle table.
    """

    evidence = {}

    datasets = {
        "components": components_df,
        "requirements": requirements_df,
        "bom": bom_df,
        "engineering_changes": changes_df,
        "design_specifications": design_df,
        "manufacturing": manufacturing_df,
        "validation": validation_df,
        "service": service_df,
        "suppliers": supplier_df,
        "relationships": relationships_df,
        "field_failures": field_df,
        "documents": documents_df,
    }

    for (
        dataset_name,
        dataset
    ) in datasets.items():

        matches = related_rows(
            dataset,
            entity_id,
        )

        if not matches.empty:

            evidence[
                dataset_name
            ] = dataframe_records(
                matches
            )

    return evidence


def keyword_search_plm(
    question,
):
    """
    Search lifecycle tables using important words from
    a question when no explicit entity ID is supplied.
    """

    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "show",
        "tell",
        "give",
        "does",
        "are",
        "the",
        "and",
        "for",
        "from",
        "with",
        "into",
        "about",
        "this",
        "that",
        "have",
        "has",
        "been",
        "aureon",
        "plm",
        "agent",
    }

    words = re.findall(
        r"[A-Za-z0-9_-]+",
        question.lower(),
    )

    keywords = [
        word
        for word in words
        if (
            len(word) >= 4
            and word not in stop_words
        )
    ]

    # Limit broad searches
    keywords = keywords[:6]

    evidence = {}

    datasets = {
        "components": components_df,
        "requirements": requirements_df,
        "bom": bom_df,
        "engineering_changes": changes_df,
        "design_specifications": design_df,
        "manufacturing": manufacturing_df,
        "validation": validation_df,
        "service": service_df,
        "suppliers": supplier_df,
        "relationships": relationships_df,
    }

    for (
        dataset_name,
        dataset
    ) in datasets.items():

        if dataset.empty:
            continue

        collected_rows = []

        for keyword in keywords:

            matches = search_dataframe_for_text(
                dataset,
                keyword,
            )

            if not matches.empty:

                collected_rows.append(
                    matches
                )

        if collected_rows:

            combined = pd.concat(
                collected_rows,
                ignore_index=True,
            ).drop_duplicates()

            # Avoid dumping huge tables into the LLM
            combined = combined.head(
                20
            )

            evidence[
                dataset_name
            ] = dataframe_records(
                combined
            )

    return evidence


def collect_question_evidence(
    question,
):
    """
    Assemble structured PLM evidence plus semantic RAG
    evidence for a user's natural-language question.
    """

    evidence = {
        "query": question,
        "identified_entities": [],
        "structured_plm_evidence": {},
        "historical_semantic_evidence": [],
    }

    entity_ids = extract_entity_ids(
        question
    )

    evidence[
        "identified_entities"
    ] = entity_ids

    # ------------------------------------------------------
    # Entity-driven deterministic retrieval
    # ------------------------------------------------------

    if entity_ids:

        for entity_id in entity_ids:

            entity_evidence = collect_entity_evidence(
                entity_id
            )

            evidence[
                "structured_plm_evidence"
            ][
                entity_id
            ] = entity_evidence

    # ------------------------------------------------------
    # Keyword retrieval for questions without IDs
    # ------------------------------------------------------

    else:

        evidence[
            "structured_plm_evidence"
        ][
            "keyword_search"
        ] = keyword_search_plm(
            question
        )

    # ------------------------------------------------------
    # Semantic RAG retrieval
    # ------------------------------------------------------

    try:

        retriever = get_retriever()

        rag_results = retriever.retrieve_evidence(
            query=question,
            top_k=5,
            exclude_ecr=None,
        )

        evidence[
            "historical_semantic_evidence"
        ] = rag_results

    except Exception as error:

        evidence[
            "historical_semantic_evidence"
        ] = {
            "retrieval_error": str(
                error
            )
        }

    return evidence


# ==========================================================
# ASK PLM AGENT — MODEL REASONING
# ==========================================================

def ask_plm_agent(
    question,
):

    client = get_openai_client()

    if client is None:

        raise RuntimeError(
            "OPENAI_API_KEY was not found in the environment."
        )

    evidence = collect_question_evidence(
        question
    )

    evidence_text = json.dumps(
        evidence,
        indent=2,
        default=str,
    )

    instructions = """
You are the Aureon AI-PLM Engineering Agent.

You support engineers and engineering students working with
the Aureon E-Move X1 Product Lifecycle Management world.

You are not a generic chatbot.

Your answer must be grounded in the supplied PLM evidence.

============================================================
EVIDENCE RULES
============================================================

Always distinguish between:

- Known from PLM data
- Explicit relationship
- Historical evidence
- Calculated value
- Engineering inference
- Unknown / missing information

Never present engineering inference as confirmed PLM fact.

A relationship between two items does not automatically mean
that either item requires redesign.

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
- manufacturing operations
- test results
- validation status
- safety conclusions
- regulatory conclusions
- requirement values
- service history
- field failures

If evidence is absent, state that the information is unknown
or requires verification.

============================================================
ANSWER STYLE
============================================================

Be clear and professional.

Use concise headings where useful.

Explain engineering significance, not merely database records.

For recommendations, use words such as:
- review
- verify
- validate
- investigate
- confirm

unless the evidence explicitly confirms an action.
"""

    user_prompt = f"""
============================================================
USER QUESTION
============================================================

{question}


============================================================
RETRIEVED AUREON PLM EVIDENCE
============================================================

{evidence_text}


============================================================
TASK
============================================================

Answer the user's question using the supplied evidence.

Where useful organize the answer as:

### Answer
### PLM Evidence
### Engineering Interpretation
### Lifecycle Relevance
### Missing / Unknown Information
### Recommended Next Step

Do not force headings that are unnecessary.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=user_prompt,
    )

    return (
        response.output_text,
        evidence,
    )


# ==========================================================
# DATA QUALITY FUNCTIONS
# ==========================================================

def unique_values(
    df,
    column_names,
):

    column = find_column(
        df,
        column_names,
    )

    if not column:
        return set()

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return set(
        values[
            values != ""
        ]
    )


def calculate_data_quality():

    checks = []

    component_ids = unique_values(
        bom_df,
        [
            "Component_ID",
            "Component ID",
        ],
    )

    requirement_ids = unique_values(
        requirements_df,
        [
            "Requirement_ID",
            "Requirement ID",
        ],
    )

    ecr_ids = unique_values(
        changes_df,
        [
            "ECR_ID",
            "ECR ID",
        ],
    )

    supplier_ids = unique_values(
        supplier_df,
        [
            "Supplier_ID",
            "Supplier ID",
        ],
    )

    # ------------------------------------------------------
    # Workbook availability
    # ------------------------------------------------------

    checks.append(
        {
            "Check": "Workbook loaded",
            "Status": "PASS",
            "Details":
                f"{len(PLM_DATA)} sheets detected",
        }
    )

    # ------------------------------------------------------
    # Core tables
    # ------------------------------------------------------

    core_tables = {
        "Requirements":
            requirements_df,

        "BOM":
            bom_df,

        "Engineering Changes":
            changes_df,

        "Design Specifications":
            design_df,

        "Manufacturing":
            manufacturing_df,

        "Validation":
            validation_df,

        "Service":
            service_df,

        "Suppliers":
            supplier_df,

        "Relationships":
            relationships_df,
    }

    for name, table in core_tables.items():

        if table.empty:

            status = "WARNING"
            details = "Sheet/table not detected"

        else:

            status = "PASS"
            details = (
                f"{len(table)} records"
            )

        checks.append(
            {
                "Check":
                    f"{name} dataset",

                "Status":
                    status,

                "Details":
                    details,
            }
        )

    # ------------------------------------------------------
    # Change primary component references
    # ------------------------------------------------------

    primary_component_col = find_column(
        changes_df,
        [
            "Primary_Component_ID",
            "Primary Component ID",
        ],
    )

    if (
        primary_component_col
        and component_ids
    ):

        referenced = set(
            changes_df[
                primary_component_col
            ]
            .dropna()
            .astype(str)
        )

        missing = (
            referenced
            - component_ids
        )

        checks.append(
            {
                "Check":
                    "ECR primary component references",

                "Status":
                    "PASS"
                    if not missing
                    else "WARNING",

                "Details":
                    "All primary components resolved"
                    if not missing
                    else (
                        "Unresolved: "
                        + ", ".join(
                            sorted(missing)
                        )
                    ),
            }
        )

    # ------------------------------------------------------
    # Validation requirement references
    # ------------------------------------------------------

    validation_req_col = find_column(
        validation_df,
        [
            "Requirement_ID",
            "Requirement ID",
        ],
    )

    if (
        validation_req_col
        and requirement_ids
    ):

        referenced = set(
            validation_df[
                validation_req_col
            ]
            .dropna()
            .astype(str)
        )

        missing = (
            referenced
            - requirement_ids
        )

        checks.append(
            {
                "Check":
                    "Validation → Requirement links",

                "Status":
                    "PASS"
                    if not missing
                    else "WARNING",

                "Details":
                    "All validation requirements resolved"
                    if not missing
                    else (
                        "Unresolved: "
                        + ", ".join(
                            sorted(missing)
                        )
                    ),
            }
        )

    return pd.DataFrame(
        checks
    )


# ==========================================================
# STUDENT LEARNING HELPERS
# ==========================================================

def get_all_entity_ids():

    entity_ids = set()

    for dataframe in [
        bom_df,
        requirements_df,
        changes_df,
        design_df,
        manufacturing_df,
        validation_df,
        service_df,
        supplier_df,
        relationships_df,
    ]:

        if dataframe.empty:
            continue

        for column in dataframe.columns:

            values = (
                dataframe[column]
                .dropna()
                .astype(str)
            )

            for value in values:

                ids = extract_entity_ids(
                    value
                )

                entity_ids.update(
                    ids
                )

    return sorted(
        entity_ids
    )


# ==========================================================
# GLOBAL CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.55rem;
        font-weight: 800;
        color: #0b1b4d;
        margin-bottom: 0.1rem;
    }

    .sub-title {
        font-size: 1.15rem;
        color: #4f5d7a;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #10214e;
        margin-top: 1.3rem;
        margin-bottom: 0.6rem;
    }

    .section-subtitle {
        font-size: 1rem;
        color: #5e6b88;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e3eaf7;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        box-shadow:
            0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #56657f;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 750;
        color: #10214e;
    }

    .feature-card {
        background: white;
        border: 1px solid #e3eaf7;
        border-radius: 14px;
        padding: 1rem;
        height: 100%;
        box-shadow:
            0 2px 6px rgba(0,0,0,0.04);
    }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #10214e;
        margin-bottom: 0.6rem;
    }

    .feature-body {
        font-size: 0.96rem;
        color: #44516b;
        line-height: 1.6;
    }

    .info-panel {
        background:
            linear-gradient(
                135deg,
                #eef6ff,
                #f8fbff
            );

        border: 1px solid #d8e6f8;
        border-radius: 16px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }

    .workflow-card {
        background: #ffffff;
        border: 1px solid #e1e9f5;
        border-radius: 14px;
        padding: 1rem;
        min-height: 135px;
    }

    .workflow-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1683ff;
    }

    .workflow-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #10214e;
        margin-bottom: 0.3rem;
    }

    .developer-box {
        border-top: 1px solid #d8dfeb;
        margin-top: 1rem;
        padding-top: 1rem;
        font-size: 0.86rem;
        color: #526078;
        line-height: 1.45;
    }

    .developer-name {
        font-weight: 750;
        color: #14254d;
        font-size: 0.94rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            use_container_width=True,
        )

    st.markdown(
        "### AI-PLM Engineering Lab"
    )

    st.caption(
        "Aureon E-Move X1"
    )

    page = st.radio(
        "Navigation",
        [
            "Home",
            "PLM World",
            "Ask PLM Agent",
            "Change Impact Analysis",
            "Data Quality",
            "Student Learning",
            "Research Evaluation",
        ],
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "**Platform Summary**"
    )

    st.write(
        "Company: **Aureon**"
    )

    st.write(
        "Product: **E-Move X1**"
    )

    st.caption(
        "Agentic Product Lifecycle Management"
    )

    # Push developer information toward bottom visually
    st.markdown(
        """
        <div style="height: 80px;"></div>

        <div class="developer-box">

        <b>Designed and Developed by:</b><br><br>

        <span class="developer-name">
        Dr. Arun C Dixit
        </span><br>

        Associate Professor, ME<br>
        VVCE, Mysuru

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# HOME
# ==========================================================

if page == "Home":

    st.markdown(
        """
        <div class="main-title">
        Aureon AI-PLM Engineering Lab
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Agentic Product Lifecycle Management
        for the Aureon E-Move X1
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_left, hero_right = st.columns(
        [1.08, 1]
    )

    with hero_left:

        st.header(
            "Aureon E-Move X1"
        )

        st.subheader(
            "Smart. Sustainable. Engineered for Tomorrow."
        )

        st.write(
            """
            The **Aureon E-Move X1** is a fictional
            next-generation electric mobility platform
            developed as a realistic engineering world
            for demonstrating Product Lifecycle Management
            and Agentic AI.

            Its PLM environment connects requirements,
            components, design specifications, BOM,
            suppliers, manufacturing, validation,
            engineering changes, field information and
            service records.
            """
        )

        st.markdown(
            "### Purpose of the AI-PLM Agent"
        )

        st.write(
            """
            The agent investigates how engineering changes
            propagate through the product lifecycle.

            It retrieves evidence, traces relationships,
            combines structured PLM information with
            relevant historical engineering context,
            distinguishes facts from inference and produces
            engineering recommendations.
            """
        )

    with hero_right:

        show_image_if_exists(
            SCOOTER_PATH,
            use_container_width=True,
        )

    m1, m2, m3, m4 = st.columns(
        4
    )

    with m1:
        metric_card(
            "Battery",
            "72 V / 40 Ah",
        )

    with m2:
        metric_card(
            "Energy",
            "2.88 kWh",
        )

    with m3:
        metric_card(
            "Motor Peak Power",
            "8 kW",
        )

    with m4:
        metric_card(
            "Range Target",
            "100 km",
        )

    st.markdown(
        """
        <div class="section-title">
        Product Architecture
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Representative major subsystems of the
        Aureon E-Move X1.
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_image_if_exists(
        ARCHITECTURE_PATH,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="section-title">
        What the Platform Contains
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(
        4
    )

    with p1:
        metric_card(
            "Components",
            len(bom_df),
        )

    with p2:
        metric_card(
            "Requirements",
            len(requirements_df),
        )

    with p3:
        metric_card(
            "Engineering Changes",
            len(changes_df),
        )

    with p4:
        metric_card(
            "Relationships",
            len(relationships_df),
        )

    p5, p6, p7, p8 = st.columns(
        4
    )

    with p5:
        metric_card(
            "Manufacturing Records",
            len(manufacturing_df),
        )

    with p6:
        metric_card(
            "Validation Tests",
            len(validation_df),
        )

    with p7:
        metric_card(
            "Service Records",
            len(service_df),
        )

    with p8:
        metric_card(
            "Suppliers",
            len(supplier_df),
        )

    st.markdown(
        """
        <div class="section-title">
        Agentic PLM Architecture
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """
Aureon PLM Dataset
        ↓
PLM Data Engine
        ↓
Deterministic PLM Tools
        ↓
Semantic RAG
        ↓
Agentic Engineering Reasoning
        ↓
Engineering Change Impact Analysis
        ↓
Evidence-Grounded Recommendation
        """,
        language="text",
    )


# ==========================================================
# PLM WORLD
# ==========================================================

elif page == "PLM World":

    st.markdown(
        """
        <div class="main-title">
        Aureon PLM World
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Explore the product lifecycle information
        of the Aureon E-Move X1.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Live PLM Dataset</b><br>

        The information on this page is loaded
        directly from the Aureon E-Move X1 PLM
        Excel database.

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:
        metric_card(
            "Components",
            len(bom_df),
        )

    with c2:
        metric_card(
            "Requirements",
            len(requirements_df),
        )

    with c3:
        metric_card(
            "Engineering Changes",
            len(changes_df),
        )

    with c4:
        metric_card(
            "Relationships",
            len(relationships_df),
        )

    c5, c6, c7, c8 = st.columns(
        4
    )

    with c5:
        metric_card(
            "Manufacturing",
            len(manufacturing_df),
        )

    with c6:
        metric_card(
            "Validation Tests",
            len(validation_df),
        )

    with c7:
        metric_card(
            "Service Records",
            len(service_df),
        )

    with c8:
        metric_card(
            "Suppliers",
            len(supplier_df),
        )

    (
        tab_overview,
        tab_components,
        tab_requirements,
        tab_bom,
        tab_changes,
        tab_design,
        tab_mfg,
        tab_validation,
        tab_service,
        tab_suppliers,
        tab_relationships,
    ) = st.tabs(
        [
            "Overview",
            "Components",
            "Requirements",
            "BOM",
            "Engineering Changes",
            "Design Specs",
            "Manufacturing",
            "Validation",
            "Service",
            "Suppliers",
            "Relationships",
        ]
    )

    # ------------------------------------------------------
    # OVERVIEW
    # ------------------------------------------------------

    with tab_overview:

        st.subheader(
            "Product Lifecycle Overview"
        )

        st.write(
            """
            Select a component to explore available
            lifecycle information connected to that
            engineering entity.
            """
        )

        component_column = find_column(
            bom_df,
            [
                "Component_ID",
                "Component ID",
            ],
        )

        if component_column:

            component_ids = (
                bom_df[
                    component_column
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            component_ids = sorted(
                component_ids
            )

        else:

            component_ids = []

        if component_ids:

            selected_component = st.selectbox(
                "Select Component",
                component_ids,
            )

            st.markdown(
                f"## Lifecycle Explorer — "
                f"{selected_component}"
            )

            explorer_datasets = {
                "BOM":
                    bom_df,

                "Design Specifications":
                    design_df,

                "Manufacturing":
                    manufacturing_df,

                "Validation":
                    validation_df,

                "Service":
                    service_df,

                "Suppliers":
                    supplier_df,

                "Relationships":
                    relationships_df,

                "Engineering Changes":
                    changes_df,
            }

            for (
                dataset_name,
                dataset
            ) in explorer_datasets.items():

                matches = related_rows(
                    dataset,
                    selected_component,
                )

                if not matches.empty:

                    with st.expander(
                        f"{dataset_name} — "
                        f"{len(matches)} record(s)",
                        expanded=False,
                    ):

                        st.dataframe(
                            matches,
                            use_container_width=True,
                            hide_index=True,
                        )

        else:

            st.warning(
                "No component IDs were detected."
            )

    with tab_components:

        st.subheader(
            "Product Components"
        )

        searchable_dataframe(
            bom_df,
            "component_search",
        )

    with tab_requirements:

        st.subheader(
            "Product Requirements"
        )

        searchable_dataframe(
            requirements_df,
            "requirements_search",
        )

    with tab_bom:

        st.subheader(
            "Bill of Materials"
        )

        searchable_dataframe(
            bom_df,
            "bom_search",
        )

    with tab_changes:

        st.subheader(
            "Engineering Change Log"
        )

        searchable_dataframe(
            changes_df,
            "changes_search",
        )

    with tab_design:

        st.subheader(
            "Design Specifications"
        )

        searchable_dataframe(
            design_df,
            "design_search",
        )

    with tab_mfg:

        st.subheader(
            "Manufacturing Routing"
        )

        searchable_dataframe(
            manufacturing_df,
            "manufacturing_search",
        )

    with tab_validation:

        st.subheader(
            "Validation Tests"
        )

        searchable_dataframe(
            validation_df,
            "validation_search",
        )

    with tab_service:

        st.subheader(
            "Service Records"
        )

        searchable_dataframe(
            service_df,
            "service_search",
        )

    with tab_suppliers:

        st.subheader(
            "Supplier Records"
        )

        searchable_dataframe(
            supplier_df,
            "supplier_search",
        )

    with tab_relationships:

        st.subheader(
            "Product Relationships"
        )

        st.write(
            """
            These relationships support engineering
            traceability and lifecycle reasoning.
            """
        )

        searchable_dataframe(
            relationships_df,
            "relationships_search",
        )


# ==========================================================
# ASK PLM AGENT
# ==========================================================

elif page == "Ask PLM Agent":

    st.markdown(
        """
        <div class="main-title">
        Ask Aureon PLM Agent
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Ask engineering questions across the
        Aureon E-Move X1 lifecycle.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Evidence-Grounded Engineering Assistant</b><br>

        The agent retrieves structured PLM information
        and relevant semantic evidence before answering.
        It separates recorded facts from engineering
        inference and missing information.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Example Questions"
    )

    e1, e2, e3 = st.columns(
        3
    )

    with e1:

        st.code(
            "What components are connected to BAT-001?",
            language=None,
        )

    with e2:

        st.code(
            "What validation evidence exists for REQ-013?",
            language=None,
        )

    with e3:

        st.code(
            "Explain the relationship between SW-001 and BMS-001.",
            language=None,
        )

    e4, e5 = st.columns(
        2
    )

    with e4:

        st.code(
            "Show manufacturing and supplier information for BAT-001.",
            language=None,
        )

    with e5:

        st.code(
            "What service issues are recorded for INV-001?",
            language=None,
        )

    question = st.text_area(
        "Ask the PLM Agent",
        height=125,
        placeholder=(
            "Example: What engineering information "
            "is available for BAT-001?"
        ),
    )

    ask_button = st.button(
        "Ask PLM Agent",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter an engineering or PLM question."
            )

        elif not OPENAI_API_KEY:

            st.error(
                "OPENAI_API_KEY is not available. "
                "Check your .env file."
            )

        else:

            with st.spinner(
                "Retrieving PLM evidence and "
                "running engineering reasoning..."
            ):

                try:

                    (
                        answer,
                        evidence,
                    ) = ask_plm_agent(
                        question
                    )

                    st.session_state[
                        "last_plm_question"
                    ] = question

                    st.session_state[
                        "last_plm_answer"
                    ] = answer

                    st.session_state[
                        "last_plm_evidence"
                    ] = evidence

                except Exception as error:

                    st.error(
                        "The PLM Agent could not "
                        "complete the request."
                    )

                    st.exception(
                        error
                    )

    if (
        "last_plm_answer"
        in st.session_state
    ):

        st.markdown(
            "---"
        )

        st.markdown(
            "## Agent Response"
        )

        st.caption(
            "Question: "
            + st.session_state.get(
                "last_plm_question",
                "",
            )
        )

        st.markdown(
            st.session_state[
                "last_plm_answer"
            ]
        )

        with st.expander(
            "View Retrieved Evidence",
            expanded=False,
        ):

            st.json(
                st.session_state[
                    "last_plm_evidence"
                ]
            )


# ==========================================================
# CHANGE IMPACT ANALYSIS
# ==========================================================

elif page == "Change Impact Analysis":

    st.markdown(
        """
        <div class="main-title">
        Engineering Change Impact Analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Investigate how an engineering change
        propagates through the product lifecycle.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Agentic Change-Impact Reasoning</b><br>

        The system combines deterministic PLM evidence,
        lifecycle relationships and semantic engineering
        context to identify likely consequences and
        recommended engineering actions.

        </div>
        """,
        unsafe_allow_html=True,
    )

    ecr_column = find_column(
        changes_df,
        [
            "ECR_ID",
            "ECR ID",
        ],
    )

    if not ecr_column:

        st.error(
            "ECR ID column could not be detected."
        )

        st.stop()

    ecr_list = (
        changes_df[
            ecr_column
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected_ecr = st.selectbox(
        "Select Engineering Change",
        ecr_list,
    )

    selected_change = related_rows(
        changes_df,
        selected_ecr,
    )

    if not selected_change.empty:

        st.markdown(
            "### Change Definition"
        )

        st.dataframe(
            selected_change,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Analysis Workflow"
    )

    w1, w2, w3 = st.columns(
        3
    )

    with w1:

        st.markdown(
            """
            <div class="workflow-card">

            <div class="workflow-number">
            01
            </div>

            <div class="workflow-title">
            Retrieve Evidence
            </div>

            Structured PLM records, relationships
            and relevant historical evidence.

            </div>
            """,
            unsafe_allow_html=True,
        )

    with w2:

        st.markdown(
            """
            <div class="workflow-card">

            <div class="workflow-number">
            02
            </div>

            <div class="workflow-title">
            Engineering Reasoning
            </div>

            Analyse components, requirements,
            configuration and lifecycle consequences.

            </div>
            """,
            unsafe_allow_html=True,
        )

    with w3:

        st.markdown(
            """
            <div class="workflow-card">

            <div class="workflow-number">
            03
            </div>

            <div class="workflow-title">
            Engineering Report
            </div>

            Separate confirmed evidence, inference,
            risk and recommended actions.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Lifecycle Propagation"
    )

    st.code(
        """
Engineering Change
        ↓
Direct Impact
        ↓
First-Order Dependencies
        ↓
Second-Order Dependencies
        ↓
Lifecycle Consequences
        ↓
Review / Verification / Validation / Action
        """,
        language="text",
    )

    run_analysis = st.button(
        "Run AI Impact Analysis",
        type="primary",
        use_container_width=True,
    )

    if run_analysis:

        with st.spinner(
            f"Analysing {selected_ecr} "
            f"across the Aureon lifecycle..."
        ):

            try:

                result = analyse_change(
                    ecr_id=selected_ecr,
                    benchmark_mode=False,
                )

                st.session_state[
                    "impact_ecr"
                ] = selected_ecr

                st.session_state[
                    "impact_result"
                ] = result

            except Exception as error:

                st.error(
                    "The engineering change-impact "
                    "analysis could not be completed."
                )

                st.exception(
                    error
                )

    if (
        "impact_result"
        in st.session_state
        and
        st.session_state.get(
            "impact_ecr"
        )
        == selected_ecr
    ):

        st.markdown(
            "---"
        )

        st.markdown(
            f"## Impact Assessment — "
            f"{selected_ecr}"
        )

        st.markdown(
            st.session_state[
                "impact_result"
            ]
        )

        st.success(
            "Engineering change-impact analysis completed."
        )

        st.caption(
            """
            The report may identify review candidates
            where evidence is incomplete. A recommended
            review is not the same as a confirmed design
            change.
            """
        )


# ==========================================================
# DATA QUALITY
# ==========================================================

elif page == "Data Quality":

    st.markdown(
        """
        <div class="main-title">
        PLM Data Quality
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Inspect dataset completeness, integrity and
        cross-lifecycle traceability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(
        4
    )

    with q1:
        metric_card(
            "Workbook Sheets",
            len(PLM_DATA),
        )

    with q2:
        metric_card(
            "BOM Components",
            len(bom_df),
        )

    with q3:
        metric_card(
            "Relationships",
            len(relationships_df),
        )

    with q4:
        metric_card(
            "Engineering Changes",
            len(changes_df),
        )

    st.markdown(
        "### Automated Integrity Review"
    )

    quality_df = calculate_data_quality()

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )

    pass_count = len(
        quality_df[
            quality_df["Status"]
            == "PASS"
        ]
    )

    warning_count = len(
        quality_df[
            quality_df["Status"]
            == "WARNING"
        ]
    )

    d1, d2 = st.columns(
        2
    )

    with d1:

        st.metric(
            "Checks Passed",
            pass_count,
        )

    with d2:

        st.metric(
            "Warnings",
            warning_count,
        )

    st.markdown(
        "### Dataset Coverage"
    )

    coverage_df = pd.DataFrame(
        {
            "Lifecycle Domain": [
                "Requirements",
                "BOM / Components",
                "Engineering Changes",
                "Design Specifications",
                "Manufacturing",
                "Validation",
                "Service",
                "Suppliers",
                "Relationships",
            ],

            "Records": [
                len(requirements_df),
                len(bom_df),
                len(changes_df),
                len(design_df),
                len(manufacturing_df),
                len(validation_df),
                len(service_df),
                len(supplier_df),
                len(relationships_df),
            ],
        }
    )

    st.bar_chart(
        coverage_df.set_index(
            "Lifecycle Domain"
        )
    )

    st.caption(
        """
        Data quality checks shown here are deterministic
        structural checks. They do not by themselves prove
        engineering correctness or completeness.
        """
    )


# ==========================================================
# STUDENT LEARNING
# ==========================================================

elif page == "Student Learning":

    st.markdown(
        """
        <div class="main-title">
        Student Learning Lab
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Learn PLM traceability, lifecycle thinking and
        engineering change reasoning through guided cases.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Learning Objective</b><br>

        Before seeing the AI response, students should
        examine an engineering change and predict which
        lifecycle entities may be affected.

        The purpose is not simply to match the AI.
        Students should justify their engineering reasoning.

        </div>
        """,
        unsafe_allow_html=True,
    )

    ecr_column = find_column(
        changes_df,
        [
            "ECR_ID",
            "ECR ID",
        ],
    )

    if ecr_column:

        learning_ecrs = (
            changes_df[
                ecr_column
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

        learning_case = st.selectbox(
            "Select Learning Case",
            learning_ecrs,
            key="learning_case",
        )

        case_record = related_rows(
            changes_df,
            learning_case,
        )

        st.markdown(
            "### Step 1 — Understand the Change"
        )

        st.dataframe(
            case_record,
            use_container_width=True,
            hide_index=True,
        )

        all_entities = get_all_entity_ids()

        st.markdown(
            "### Step 2 — Predict the Impact"
        )

        student_entities = st.multiselect(
            "Which entities do you think may be affected?",
            all_entities,
            key="student_entities",
        )

        student_reasoning = st.text_area(
            "Explain your engineering reasoning",
            height=140,
            placeholder=(
                "Why do you expect these components, "
                "requirements or lifecycle records to "
                "be affected?"
            ),
        )

        if st.button(
            "Save My Prediction",
            use_container_width=True,
        ):

            st.session_state[
                "student_prediction"
            ] = {
                "ecr":
                    learning_case,

                "entities":
                    student_entities,

                "reasoning":
                    student_reasoning,
            }

            st.success(
                "Student prediction recorded for this session."
            )

        st.markdown(
            "### Step 3 — Compare with the AI"
        )

        if st.button(
            "Run Agent for Learning Comparison",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Running evidence-grounded change analysis..."
            ):

                try:

                    learning_result = analyse_change(
                        ecr_id=learning_case,
                        benchmark_mode=False,
                    )

                    st.session_state[
                        "learning_agent_result"
                    ] = learning_result

                    st.session_state[
                        "learning_agent_ecr"
                    ] = learning_case

                except Exception as error:

                    st.error(
                        "The learning comparison could not be completed."
                    )

                    st.exception(
                        error
                    )

        if (
            st.session_state.get(
                "learning_agent_ecr"
            )
            == learning_case
            and
            "learning_agent_result"
            in st.session_state
        ):

            student_prediction = (
                st.session_state.get(
                    "student_prediction",
                    {}
                )
            )

            if (
                student_prediction.get(
                    "ecr"
                )
                == learning_case
            ):

                with st.expander(
                    "Your Prediction",
                    expanded=True,
                ):

                    st.write(
                        "**Selected entities:**"
                    )

                    st.write(
                        student_prediction.get(
                            "entities",
                            [],
                        )
                    )

                    st.write(
                        "**Reasoning:**"
                    )

                    st.write(
                        student_prediction.get(
                            "reasoning",
                            "",
                        )
                    )

            st.markdown(
                "#### AI Engineering Assessment"
            )

            st.markdown(
                st.session_state[
                    "learning_agent_result"
                ]
            )

            st.info(
                """
                Reflect on why the AI included or excluded
                lifecycle areas. Differences should be
                examined using evidence rather than treated
                automatically as student mistakes.
                """
            )

    else:

        st.warning(
            "Engineering change records were not detected."
        )


# ==========================================================
# RESEARCH EVALUATION
# ==========================================================

elif page == "Research Evaluation":

    st.markdown(
        """
        <div class="main-title">
        Research Evaluation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Evaluation of autonomous AI-PLM engineering
        impact reasoning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Research Question</b><br>

        Can an agentic AI system perform
        evidence-grounded engineering-change impact
        assessment across interconnected PLM lifecycle
        data while supporting PLM learning and
        engineering judgement?

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### D1 Held-Out Benchmark"
    )

    r1, r2, r3 = st.columns(
        3
    )

    with r1:

        st.metric(
            "Macro Precision",
            "0.5455",
        )

    with r2:

        st.metric(
            "Macro Recall",
            "0.5471",
        )

    with r3:

        st.metric(
            "Macro F1",
            "0.5325",
        )

    r4, r5, r6 = st.columns(
        3
    )

    with r4:

        st.metric(
            "Micro Precision",
            "0.5119",
        )

    with r5:

        st.metric(
            "Micro Recall",
            "0.5309",
        )

    with r6:

        st.metric(
            "Micro F1",
            "0.5212",
        )

    st.caption(
        """
        Held-out D1 benchmark: nine engineering-change
        cases. ECR-001 was used as the development/pilot
        case and was excluded from this aggregate.
        """
    )

    st.markdown(
        "### Prospective Pilot — ECR-002"
    )

    comparison_df = pd.DataFrame(
        {
            "System": [
                "D1",
                "D2.2",
            ],

            "Precision": [
                0.6154,
                0.6875,
            ],

            "Recall": [
                0.6154,
                0.8462,
            ],

            "F1": [
                0.6154,
                0.7586,
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        comparison_df.set_index(
            "System"
        )[
            [
                "Precision",
                "Recall",
                "F1",
            ]
        ]
    )

    st.markdown(
        "### Computational Comparison"
    )

    efficiency_df = pd.DataFrame(
        {
            "System": [
                "D1",
                "D2.2",
            ],

            "PLM Tool Calls": [
                22,
                23,
            ],

            "Model Tokens": [
                27106,
                47814,
            ],

            "Elapsed Time (s)": [
                54.24,
                98.60,
            ],
        }
    )

    st.dataframe(
        efficiency_df,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        """
        ECR-002 is a single prospective pilot comparison.
        It must not be presented as a completed five-case
        prospective benchmark or as statistically conclusive
        evidence of superiority.
        """
    )

    st.markdown(
        "### Observed Error Pattern"
    )

    st.write(
        """
        Earlier benchmark analysis indicated two important
        reasoning challenges:

        **Under-propagation** — missing some downstream
        component, requirement, validation or service
        consequences.

        **Over-adjudication** — promoting contextually
        related records into predicted impacts without
        sufficient evidence.

        The later D2 architectures were developed to
        investigate these limitations through lifecycle
        completeness review and evidence-based candidate
        adjudication.
        """
    )

    st.markdown(
        "### Evidence-Scoring Concept"
    )

    scoring_df = pd.DataFrame(
        {
            "Evidence Type": [
                "Direct change evidence",
                "Explicit relationship",
                "Engineering consequence",
                "Requirement / validation support",
                "Lifecycle action",
                "Context-only evidence",
            ],

            "D2.2 Score": [
                "+3",
                "+2",
                "+2",
                "+1",
                "+1",
                "-2",
            ],
        }
    )

    st.dataframe(
        scoring_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        """
        The research evaluation page reports previously
        completed experimental results. It does not run
        benchmark ground truth inside the production
        PLM Agent.
        """
    )