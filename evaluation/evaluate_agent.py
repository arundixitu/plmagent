import json
import csv
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

EVALUATION_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
)

PREDICTIONS_FOLDER = (
    EVALUATION_FOLDER
    / "predictions"
)

RESULTS_FOLDER = (
    EVALUATION_FOLDER
    / "results"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


GROUND_TRUTH_FILE = (
    DATA_FOLDER
    / "99_Ground_Truth_Impact_Matrix.xlsx"
)


# ==========================================================
# EXPECTED GROUND-TRUTH SCHEMA
# ==========================================================

GT_ECR_COLUMN = "ECR_ID"

GT_ENTITY_COLUMN = (
    "Expected_Impacted_Entities"
)

GT_RISK_COLUMN = "Risk_Level"


# ==========================================================
# UTILITY
# ==========================================================

def normalize_entity_id(
    value
):
    """
    Normalize one entity identifier.
    """

    if value is None:
        return None

    entity = str(
        value
    ).strip().upper()

    if not entity:
        return None

    if entity in {
        "NAN",
        "NONE",
        "NULL"
    }:
        return None

    return entity


# ==========================================================
# GROUND-TRUTH ENTITY PARSER
# ==========================================================

def parse_entity_list(
    value
):
    """
    Parse Expected_Impacted_Entities safely.

    Supports:
    - comma-separated IDs
    - semicolon-separated IDs
    - pipe-separated IDs
    - newline-separated IDs
    - Python/JSON-like list strings
    """

    if value is None:
        return []


    # ------------------------------------------------------
    # ALREADY A COLLECTION
    # ------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
            set
        )
    ):

        entities = []

        for item in value:

            entity = normalize_entity_id(
                item
            )

            if entity:
                entities.append(
                    entity
                )

        return sorted(
            set(entities)
        )


    text = str(
        value
    ).strip()


    if not text:
        return []


    if text.lower() == "nan":
        return []


    # ------------------------------------------------------
    # TRY JSON / PYTHON-LIKE LIST
    # ------------------------------------------------------

    if (
        text.startswith("[")
        and text.endswith("]")
    ):

        cleaned = text.replace(
            "'",
            '"'
        )

        try:

            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                list
            ):

                entities = []

                for item in parsed:

                    entity = normalize_entity_id(
                        item
                    )

                    if entity:
                        entities.append(
                            entity
                        )

                return sorted(
                    set(entities)
                )

        except json.JSONDecodeError:
            pass


    # ------------------------------------------------------
    # GENERIC DELIMITER PARSING
    # ------------------------------------------------------

    parts = re.split(
        r"[,;\|\n]+",
        text
    )


    entities = []

    for part in parts:

        entity = normalize_entity_id(
            part
        )

        if entity:
            entities.append(
                entity
            )


    return sorted(
        set(entities)
    )


# ==========================================================
# FIND LATEST PREDICTION
# ==========================================================

def find_latest_prediction(
    ecr_id,
    benchmark_only=True
):
    """
    Find the newest saved prediction file for an ECR.
    """

    ecr_id = normalize_entity_id(
        ecr_id
    )


    if benchmark_only:

        pattern = (
            f"{ecr_id}_benchmark_*.json"
        )

    else:

        pattern = (
            f"{ecr_id}_*.json"
        )


    files = list(
        PREDICTIONS_FOLDER.glob(
            pattern
        )
    )


    if not files:

        raise FileNotFoundError(
            (
                "No saved prediction found for "
                f"{ecr_id} in:\n"
                f"{PREDICTIONS_FOLDER}"
            )
        )


    latest_file = max(
        files,
        key=lambda path:
            path.stat().st_mtime
    )


    return latest_file


# ==========================================================
# LOAD PREDICTION
# ==========================================================

def load_prediction(
    prediction_file
):
    """
    Load one saved prediction JSON.
    """

    with open(
        prediction_file,
        "r",
        encoding="utf-8"
    ) as file:

        payload = json.load(
            file
        )


    prediction = payload.get(
        "prediction",
        {}
    )


    predicted_impacts = (
        prediction.get(
            "predicted_impacts",
            []
        )
    )


    predicted_entities = []


    for impact in predicted_impacts:

        if not isinstance(
            impact,
            dict
        ):
            continue

        entity_id = normalize_entity_id(
            impact.get(
                "entity_id"
            )
        )

        if entity_id:

            predicted_entities.append(
                entity_id
            )


    predicted_entities = sorted(
        set(
            predicted_entities
        )
    )


    return {
        "payload":
            payload,

        "prediction":
            prediction,

        "predicted_impacts":
            predicted_impacts,

        "predicted_entities":
            predicted_entities
    }


# ==========================================================
# LOAD GROUND TRUTH
# ==========================================================

def load_ground_truth(
    ecr_id
):
    """
    Load evaluator-only ground truth for one ECR.

    IMPORTANT:
    This function exists only in the evaluation layer.
    It must never be imported into the production agent.
    """

    if not GROUND_TRUTH_FILE.exists():

        raise FileNotFoundError(
            (
                "Ground-truth workbook not found:\n"
                f"{GROUND_TRUTH_FILE}"
            )
        )


    df = pd.read_excel(
        GROUND_TRUTH_FILE
    )


    required_columns = {
        GT_ECR_COLUMN,
        GT_ENTITY_COLUMN,
        GT_RISK_COLUMN
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            (
                "Ground-truth workbook is missing "
                "required columns: "
                f"{sorted(missing_columns)}"
            )
        )


    ecr_id = normalize_entity_id(
        ecr_id
    )


    matching = df[
        df[
            GT_ECR_COLUMN
        ]
        .astype(str)
        .str.upper()
        .str.strip()
        == ecr_id
    ]


    if matching.empty:

        raise ValueError(
            (
                "No evaluator ground truth exists "
                f"for {ecr_id}."
            )
        )


    if len(matching) > 1:

        raise ValueError(
            (
                "Multiple ground-truth rows found "
                f"for {ecr_id}. "
                "Expected exactly one."
            )
        )


    row = matching.iloc[0]


    expected_entities = (
        parse_entity_list(
            row[
                GT_ENTITY_COLUMN
            ]
        )
    )


    risk_level = str(
        row[
            GT_RISK_COLUMN
        ]
    ).strip()


    gt_id = None

    if "GT_ID" in df.columns:

        gt_id = str(
            row[
                "GT_ID"
            ]
        ).strip()


    return {
        "gt_id":
            gt_id,

        "ecr_id":
            ecr_id,

        "expected_entities":
            expected_entities,

        "risk_level":
            risk_level
    }


# ==========================================================
# CALCULATE METRICS
# ==========================================================

def calculate_entity_metrics(
    predicted_entities,
    expected_entities
):
    """
    Compute exact entity-level benchmark metrics.
    """

    predicted_set = set(
        predicted_entities
    )

    expected_set = set(
        expected_entities
    )


    true_positives = sorted(
        predicted_set
        & expected_set
    )


    false_positives = sorted(
        predicted_set
        - expected_set
    )


    false_negatives = sorted(
        expected_set
        - predicted_set
    )


    tp = len(
        true_positives
    )

    fp = len(
        false_positives
    )

    fn = len(
        false_negatives
    )


    # ------------------------------------------------------
    # PRECISION
    # ------------------------------------------------------

    if (
        tp + fp
    ) == 0:

        precision = 0.0

    else:

        precision = (
            tp
            / (
                tp + fp
            )
        )


    # ------------------------------------------------------
    # RECALL
    # ------------------------------------------------------

    if (
        tp + fn
    ) == 0:

        recall = 0.0

    else:

        recall = (
            tp
            / (
                tp + fn
            )
        )


    # ------------------------------------------------------
    # F1
    # ------------------------------------------------------

    if (
        precision
        + recall
    ) == 0:

        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )


    return {
        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_positive_entities":
            true_positives,

        "false_positive_entities":
            false_positives,

        "false_negative_entities":
            false_negatives
    }


# ==========================================================
# OPTIONAL RISK COMPARISON
# ==========================================================

def extract_agent_risk(
    prediction_payload
):
    """
    Risk is not yet part of the machine-readable prediction
    schema, so this currently returns None.

    The function is included for future Phase 5C extensions.
    """

    return None


# ==========================================================
# SAVE DETAILED JSON RESULT
# ==========================================================

def save_json_result(
    result
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    ecr_id = result[
        "ecr_id"
    ]


    filename = (
        f"{ecr_id}_"
        f"evaluation_"
        f"{timestamp}.json"
    )


    output_path = (
        RESULTS_FOLDER
        / filename
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )


    return output_path


# ==========================================================
# APPEND SUMMARY CSV
# ==========================================================

def append_summary_csv(
    result
):
    """
    Append one experimental run to a cumulative CSV.

    This will later support comparison across ECRs and
    across experimental systems.
    """

    csv_path = (
        RESULTS_FOLDER
        / "agent_evaluation_summary.csv"
    )


    file_exists = (
        csv_path.exists()
    )


    metrics = result[
        "entity_metrics"
    ]


    execution = result.get(
        "agent_execution_metrics",
        {}
    )


    row = {

        "Evaluation_Timestamp":
            result[
                "evaluated_at"
            ],

        "ECR_ID":
            result[
                "ecr_id"
            ],

        "GT_ID":
            result.get(
                "gt_id"
            ),

        "Ground_Truth_Risk":
            result.get(
                "ground_truth_risk"
            ),

        "Prediction_File":
            result[
                "prediction_file"
            ],

        "Predicted_Count":
            result[
                "predicted_entity_count"
            ],

        "Ground_Truth_Count":
            result[
                "ground_truth_entity_count"
            ],

        "TP":
            metrics[
                "tp"
            ],

        "FP":
            metrics[
                "fp"
            ],

        "FN":
            metrics[
                "fn"
            ],

        "Precision":
            round(
                metrics[
                    "precision"
                ],
                6
            ),

        "Recall":
            round(
                metrics[
                    "recall"
                ],
                6
            ),

        "F1":
            round(
                metrics[
                    "f1"
                ],
                6
            ),

        "Reasoning_Cycles":
            execution.get(
                "reasoning_cycles"
            ),

        "Tool_Calls_Requested":
            execution.get(
                "tool_calls_requested"
            ),

        "Tool_Calls_Executed":
            execution.get(
                "tool_calls_executed"
            ),

        "Input_Tokens":
            execution.get(
                "input_tokens"
            ),

        "Output_Tokens":
            execution.get(
                "output_tokens"
            ),

        "Total_Tokens":
            execution.get(
                "total_tokens"
            ),

        "Elapsed_Seconds":
            execution.get(
                "elapsed_seconds"
            )
    }


    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            )
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow(
            row
        )


    return csv_path


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def print_entity_list(
    title,
    entities
):

    print(
        f"\n{title} "
        f"({len(entities)}):"
    )


    if not entities:

        print(
            "  None"
        )

        return


    for entity in entities:

        print(
            f"  - {entity}"
        )


# ==========================================================
# EVALUATE ONE ECR
# ==========================================================

def evaluate_ecr(
    ecr_id,
    prediction_file=None
):
    """
    Evaluate one saved benchmark prediction against the
    evaluator-only ground truth.
    """

    ecr_id = normalize_entity_id(
        ecr_id
    )


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    if prediction_file is None:

        prediction_file = (
            find_latest_prediction(
                ecr_id=ecr_id,
                benchmark_only=True
            )
        )


    prediction_file = Path(
        prediction_file
    )


    prediction_data = (
        load_prediction(
            prediction_file
        )
    )


    payload = prediction_data[
        "payload"
    ]


    prediction_ecr = normalize_entity_id(
        payload.get(
            "ecr_id"
        )
    )


    if prediction_ecr != ecr_id:

        raise ValueError(
            (
                "Prediction file ECR mismatch. "
                f"Requested {ecr_id}, but file "
                f"contains {prediction_ecr}."
            )
        )


    if not payload.get(
        "benchmark_mode",
        False
    ):

        raise ValueError(
            (
                "Selected prediction was not generated "
                "in benchmark-safe mode. "
                "Evaluation aborted."
            )
        )


    predicted_entities = (
        prediction_data[
            "predicted_entities"
        ]
    )


    # ------------------------------------------------------
    # HIDDEN GROUND TRUTH
    # ------------------------------------------------------

    ground_truth = (
        load_ground_truth(
            ecr_id
        )
    )


    expected_entities = (
        ground_truth[
            "expected_entities"
        ]
    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    entity_metrics = (
        calculate_entity_metrics(
            predicted_entities=
                predicted_entities,

            expected_entities=
                expected_entities
        )
    )


    # ------------------------------------------------------
    # RESULT PACKAGE
    # ------------------------------------------------------

    result = {

        "evaluation_system":
            "Aureon PLM Agent Ground-Truth Evaluator",

        "evaluation_version":
            "Phase 5C.1",

        "evaluated_at":
            datetime.now().isoformat(),

        "ecr_id":
            ecr_id,

        "gt_id":
            ground_truth[
                "gt_id"
            ],

        "ground_truth_risk":
            ground_truth[
                "risk_level"
            ],

        "prediction_file":
            str(
                prediction_file
            ),

        "benchmark_mode":
            True,

        "predicted_entity_count":
            len(
                predicted_entities
            ),

        "ground_truth_entity_count":
            len(
                expected_entities
            ),

        "predicted_entities":
            predicted_entities,

        "ground_truth_entities":
            expected_entities,

        "entity_metrics":
            entity_metrics,

        "agent_execution_metrics":
            payload.get(
                "metrics",
                {}
            )
    }


    return result


# ==========================================================
# PRINT EVALUATION REPORT
# ==========================================================

def print_evaluation_report(
    result
):

    metrics = result[
        "entity_metrics"
    ]


    print(
        "\n"
        + "=" * 90
    )

    print(
        "AUREON AUTONOMOUS AI-PLM AGENT"
    )

    print(
        "GROUND-TRUTH BENCHMARK EVALUATION"
    )

    print(
        "=" * 90
    )


    print(
        f"\nECR: "
        f"{result['ecr_id']}"
    )


    print(
        f"Ground Truth ID: "
        f"{result['gt_id']}"
    )


    print(
        f"Ground Truth Risk: "
        f"{result['ground_truth_risk']}"
    )


    print(
        "\nPrediction file:"
    )

    print(
        result[
            "prediction_file"
        ]
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "ENTITY-LEVEL RESULTS"
    )

    print(
        "-" * 90
    )


    print(
        f"\nPredicted entities: "
        f"{result['predicted_entity_count']}"
    )


    print(
        f"Ground-truth entities: "
        f"{result['ground_truth_entity_count']}"
    )


    print(
        f"\nTrue Positives (TP): "
        f"{metrics['tp']}"
    )


    print(
        f"False Positives (FP): "
        f"{metrics['fp']}"
    )


    print(
        f"False Negatives (FN): "
        f"{metrics['fn']}"
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "PRIMARY METRICS"
    )

    print(
        "-" * 90
    )


    print(
        f"\nPrecision: "
        f"{metrics['precision']:.4f}"
    )


    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )


    print(
        f"F1 Score:  "
        f"{metrics['f1']:.4f}"
    )


    print_entity_list(
        "TRUE POSITIVE ENTITIES",
        metrics[
            "true_positive_entities"
        ]
    )


    print_entity_list(
        "FALSE POSITIVE ENTITIES",
        metrics[
            "false_positive_entities"
        ]
    )


    print_entity_list(
        "MISSED GROUND-TRUTH ENTITIES",
        metrics[
            "false_negative_entities"
        ]
    )


    execution = result.get(
        "agent_execution_metrics",
        {}
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "AGENT EFFICIENCY"
    )

    print(
        "-" * 90
    )


    print(
        f"\nReasoning cycles: "
        f"{execution.get('reasoning_cycles', 'N/A')}"
    )


    print(
        f"Tool requests: "
        f"{execution.get('tool_calls_requested', 'N/A')}"
    )


    print(
        f"Tools executed: "
        f"{execution.get('tool_calls_executed', 'N/A')}"
    )


    print(
        f"Total tokens: "
        f"{execution.get('total_tokens', 'N/A')}"
    )


    print(
        f"Elapsed seconds: "
        f"{execution.get('elapsed_seconds', 'N/A')}"
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
        "AUREON PLM AGENT EVALUATOR"
    )

    print(
        "Phase 5C.1 - Hidden Ground-Truth Evaluation"
    )

    print(
        "=" * 90
    )


    ecr_id = input(
        "\nEnter ECR ID "
        "(example ECR-001): "
    ).strip().upper()


    try:

        result = evaluate_ecr(
            ecr_id=ecr_id
        )


        print_evaluation_report(
            result
        )


        json_path = (
            save_json_result(
                result
            )
        )


        csv_path = (
            append_summary_csv(
                result
            )
        )


        print(
            "\n"
            + "=" * 90
        )

        print(
            "EVALUATION SAVED"
        )

        print(
            "=" * 90
        )


        print(
            "\nDetailed JSON:"
        )

        print(
            json_path
        )


        print(
            "\nCumulative CSV:"
        )

        print(
            csv_path
        )


        print(
            "\n"
            + "=" * 90
        )

        print(
            "EVALUATION COMPLETE"
        )

        print(
            "=" * 90
        )


    except Exception as error:

        print(
            "\n"
            + "=" * 90
        )

        print(
            "EVALUATION FAILED"
        )

        print(
            "=" * 90
        )


        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )


        print(
            "\nNo evaluation result was saved."
        )


        sys.exit(
            1
        )