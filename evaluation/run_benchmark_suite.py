import csv
import json
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path


# ==========================================================
# PROJECT ROOT / PYTHON PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORT FROZEN AGENT
# ==========================================================

from agent.tool_calling_agent import (
    analyse_change_agentically,
    extract_prediction_json,
    validate_prediction,
    save_prediction,
    MODEL
)


# ==========================================================
# IMPORT EVALUATOR
# ==========================================================

from evaluation.evaluate_agent import (
    evaluate_ecr,
    save_json_result,
    append_summary_csv
)


# ==========================================================
# BENCHMARK CONFIGURATION
# ==========================================================

# ECR-001 is deliberately excluded.
#
# It was used during system development and architecture
# validation and therefore must not be included in the
# headline held-out benchmark.

BENCHMARK_ECRS = [
    "ECR-004",
    "ECR-005",
    "ECR-006",
    "ECR-007",
    "ECR-008",
    "ECR-009",
    "ECR-013",
    "ECR-015",
    "ECR-016"
]


DEVELOPMENT_ECR = "ECR-001"


# ==========================================================
# PATHS
# ==========================================================

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


AGENT_FILE = (
    PROJECT_ROOT
    / "agent"
    / "tool_calling_agent.py"
)


# ==========================================================
# FILE HASHING
# ==========================================================

def calculate_sha256(
    file_path
):
    """
    Calculate SHA-256 fingerprint of a file.

    This lets us prove that the same frozen agent was used
    throughout the held-out benchmark suite.
    """

    sha256 = hashlib.sha256()


    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                8192
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )


    return sha256.hexdigest()


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def print_separator(
    character="=",
    length=90
):

    print(
        character * length
    )


def print_entity_list(
    title,
    entities
):

    print(
        f"\n{title} ({len(entities)}):"
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
# RUN ONE BENCHMARK CASE
# ==========================================================

def run_single_case(
    ecr_id,
    frozen_agent_hash
):
    """
    Run one held-out ECR through:

    1. benchmark-safe autonomous agent
    2. structured prediction extraction
    3. prediction validation
    4. prediction save
    5. hidden ground-truth evaluation

    Ground truth is accessed only after the agent has
    completed and the prediction has been saved.
    """

    # ------------------------------------------------------
    # VERIFY AGENT HAS NOT CHANGED
    # ------------------------------------------------------

    current_hash = calculate_sha256(
        AGENT_FILE
    )


    if current_hash != frozen_agent_hash:

        raise RuntimeError(
            (
                "Frozen agent file changed during the "
                "benchmark suite.\n"
                "Benchmark aborted to preserve experimental "
                "integrity."
            )
        )


    print(
        "\n"
    )

    print_separator()

    print(
        f"HELD-OUT BENCHMARK CASE: {ecr_id}"
    )

    print_separator()


    print(
        "\nRunning autonomous agent in "
        "BENCHMARK-SAFE mode..."
    )


    case_start = time.perf_counter()


    # ======================================================
    # STAGE 1 — AGENT PREDICTION
    # ======================================================

    agent_result = (
        analyse_change_agentically(
            ecr_id=ecr_id,
            benchmark_mode=True
        )
    )


    # ======================================================
    # STAGE 2 — EXTRACT STRUCTURED PREDICTION
    # ======================================================

    prediction = (
        extract_prediction_json(
            agent_result[
                "answer"
            ]
        )
    )


    prediction_errors = (
        validate_prediction(
            prediction,
            expected_ecr_id=ecr_id
        )
    )


    if prediction_errors:

        print(
            "\nSTRUCTURED PREDICTION FAILED"
        )

        for error in prediction_errors:

            print(
                f"- {error}"
            )


        return {
            "ecr_id":
                ecr_id,

            "status":
                "prediction_failed",

            "errors":
                prediction_errors,

            "agent_metrics":
                agent_result.get(
                    "metrics",
                    {}
                )
        }


    # ======================================================
    # STAGE 3 — SAVE PREDICTION
    # ======================================================

    prediction_file = (
        save_prediction(
            prediction=prediction,
            ecr_id=ecr_id,
            benchmark_mode=True,
            metrics=agent_result[
                "metrics"
            ]
        )
    )


    print(
        "\nStructured prediction saved:"
    )

    print(
        prediction_file
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
        f"\nPredicted entity count: "
        f"{len(predicted_entities)}"
    )


    # ======================================================
    # SECURITY BOUNDARY
    # ======================================================

    print(
        "\nAgent phase complete."
    )

    print(
        "Prediction frozen."
    )

    print(
        "Ground truth will now be opened by the evaluator."
    )


    # ======================================================
    # STAGE 4 — HIDDEN GROUND-TRUTH EVALUATION
    # ======================================================

    evaluation_result = (
        evaluate_ecr(
            ecr_id=ecr_id,
            prediction_file=prediction_file
        )
    )


    # Save individual evaluation JSON.
    evaluation_json = (
        save_json_result(
            evaluation_result
        )
    )


    # Append to existing cumulative evaluator CSV.
    append_summary_csv(
        evaluation_result
    )


    metrics = (
        evaluation_result[
            "entity_metrics"
        ]
    )


    case_elapsed = round(
        time.perf_counter()
        - case_start,
        3
    )


    print(
        "\n"
    )

    print_separator(
        "-"
    )

    print(
        f"{ecr_id} RESULTS"
    )

    print_separator(
        "-"
    )


    print(
        f"\nTP: {metrics['tp']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
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
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )


    print_entity_list(
        "Correct predictions",
        metrics[
            "true_positive_entities"
        ]
    )


    print_entity_list(
        "Extra predictions",
        metrics[
            "false_positive_entities"
        ]
    )


    print_entity_list(
        "Missed ground-truth entities",
        metrics[
            "false_negative_entities"
        ]
    )


    print(
        f"\nEvaluation JSON:"
    )

    print(
        evaluation_json
    )


    # ======================================================
    # CASE RESULT
    # ======================================================

    return {

        "ecr_id":
            ecr_id,

        "status":
            "success",

        "prediction_file":
            str(
                prediction_file
            ),

        "evaluation_file":
            str(
                evaluation_json
            ),

        "predicted_count":
            evaluation_result[
                "predicted_entity_count"
            ],

        "ground_truth_count":
            evaluation_result[
                "ground_truth_entity_count"
            ],

        "tp":
            metrics[
                "tp"
            ],

        "fp":
            metrics[
                "fp"
            ],

        "fn":
            metrics[
                "fn"
            ],

        "precision":
            metrics[
                "precision"
            ],

        "recall":
            metrics[
                "recall"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "true_positive_entities":
            metrics[
                "true_positive_entities"
            ],

        "false_positive_entities":
            metrics[
                "false_positive_entities"
            ],

        "false_negative_entities":
            metrics[
                "false_negative_entities"
            ],

        "agent_metrics":
            evaluation_result.get(
                "agent_execution_metrics",
                {}
            ),

        "case_elapsed_seconds":
            case_elapsed
    }


# ==========================================================
# AGGREGATE METRICS
# ==========================================================

def calculate_aggregate_metrics(
    case_results
):
    """
    Calculate macro and micro entity-level metrics.
    """

    successful = [
        result
        for result in case_results
        if result.get(
            "status"
        ) == "success"
    ]


    if not successful:

        return {
            "successful_cases": 0,
            "failed_cases": len(
                case_results
            )
        }


    # ======================================================
    # MACRO METRICS
    # ======================================================

    macro_precision = sum(
        result[
            "precision"
        ]
        for result in successful
    ) / len(
        successful
    )


    macro_recall = sum(
        result[
            "recall"
        ]
        for result in successful
    ) / len(
        successful
    )


    macro_f1 = sum(
        result[
            "f1"
        ]
        for result in successful
    ) / len(
        successful
    )


    # ======================================================
    # MICRO METRICS
    # ======================================================

    total_tp = sum(
        result[
            "tp"
        ]
        for result in successful
    )


    total_fp = sum(
        result[
            "fp"
        ]
        for result in successful
    )


    total_fn = sum(
        result[
            "fn"
        ]
        for result in successful
    )


    if (
        total_tp
        + total_fp
    ) == 0:

        micro_precision = 0.0

    else:

        micro_precision = (
            total_tp
            / (
                total_tp
                + total_fp
            )
        )


    if (
        total_tp
        + total_fn
    ) == 0:

        micro_recall = 0.0

    else:

        micro_recall = (
            total_tp
            / (
                total_tp
                + total_fn
            )
        )


    if (
        micro_precision
        + micro_recall
    ) == 0:

        micro_f1 = 0.0

    else:

        micro_f1 = (
            2
            * micro_precision
            * micro_recall
            / (
                micro_precision
                + micro_recall
            )
        )


    # ======================================================
    # EFFICIENCY METRICS
    # ======================================================

    tool_calls = []

    reasoning_cycles = []

    total_tokens = []

    runtimes = []


    for result in successful:

        agent_metrics = result.get(
            "agent_metrics",
            {}
        )


        tool_value = agent_metrics.get(
            "tool_calls_executed"
        )

        if isinstance(
            tool_value,
            (
                int,
                float
            )
        ):

            tool_calls.append(
                tool_value
            )


        cycle_value = agent_metrics.get(
            "reasoning_cycles"
        )

        if isinstance(
            cycle_value,
            (
                int,
                float
            )
        ):

            reasoning_cycles.append(
                cycle_value
            )


        token_value = agent_metrics.get(
            "total_tokens"
        )

        if isinstance(
            token_value,
            (
                int,
                float
            )
        ):

            total_tokens.append(
                token_value
            )


        runtime_value = agent_metrics.get(
            "elapsed_seconds"
        )

        if isinstance(
            runtime_value,
            (
                int,
                float
            )
        ):

            runtimes.append(
                runtime_value
            )


    def safe_mean(
        values
    ):

        if not values:
            return None

        return (
            sum(values)
            / len(values)
        )


    return {

        "successful_cases":
            len(
                successful
            ),

        "failed_cases":
            len(
                case_results
            )
            - len(
                successful
            ),

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "micro_tp":
            total_tp,

        "micro_fp":
            total_fp,

        "micro_fn":
            total_fn,

        "micro_precision":
            micro_precision,

        "micro_recall":
            micro_recall,

        "micro_f1":
            micro_f1,

        "mean_tool_calls":
            safe_mean(
                tool_calls
            ),

        "mean_reasoning_cycles":
            safe_mean(
                reasoning_cycles
            ),

        "mean_total_tokens":
            safe_mean(
                total_tokens
            ),

        "mean_runtime_seconds":
            safe_mean(
                runtimes
            )
    }


# ==========================================================
# SAVE SUITE JSON
# ==========================================================

def save_suite_json(
    suite_result
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output_path = (
        RESULTS_FOLDER
        / (
            "heldout_benchmark_suite_"
            f"{timestamp}.json"
        )
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            suite_result,
            file,
            indent=2,
            ensure_ascii=False
        )


    return output_path


# ==========================================================
# SAVE SUITE CSV
# ==========================================================

def save_suite_csv(
    case_results
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output_path = (
        RESULTS_FOLDER
        / (
            "heldout_benchmark_per_ecr_"
            f"{timestamp}.csv"
        )
    )


    columns = [
        "ECR_ID",
        "Status",
        "Predicted_Count",
        "Ground_Truth_Count",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Reasoning_Cycles",
        "Tool_Calls",
        "Total_Tokens",
        "Elapsed_Seconds"
    ]


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )


        writer.writeheader()


        for result in case_results:

            agent_metrics = (
                result.get(
                    "agent_metrics",
                    {}
                )
            )


            writer.writerow(
                {

                    "ECR_ID":
                        result.get(
                            "ecr_id"
                        ),

                    "Status":
                        result.get(
                            "status"
                        ),

                    "Predicted_Count":
                        result.get(
                            "predicted_count"
                        ),

                    "Ground_Truth_Count":
                        result.get(
                            "ground_truth_count"
                        ),

                    "TP":
                        result.get(
                            "tp"
                        ),

                    "FP":
                        result.get(
                            "fp"
                        ),

                    "FN":
                        result.get(
                            "fn"
                        ),

                    "Precision":
                        (
                            round(
                                result[
                                    "precision"
                                ],
                                6
                            )
                            if result.get(
                                "precision"
                            ) is not None
                            else None
                        ),

                    "Recall":
                        (
                            round(
                                result[
                                    "recall"
                                ],
                                6
                            )
                            if result.get(
                                "recall"
                            ) is not None
                            else None
                        ),

                    "F1":
                        (
                            round(
                                result[
                                    "f1"
                                ],
                                6
                            )
                            if result.get(
                                "f1"
                            ) is not None
                            else None
                        ),

                    "Reasoning_Cycles":
                        agent_metrics.get(
                            "reasoning_cycles"
                        ),

                    "Tool_Calls":
                        agent_metrics.get(
                            "tool_calls_executed"
                        ),

                    "Total_Tokens":
                        agent_metrics.get(
                            "total_tokens"
                        ),

                    "Elapsed_Seconds":
                        agent_metrics.get(
                            "elapsed_seconds"
                        )
                }
            )


    return output_path


# ==========================================================
# PRINT FINAL SUITE SUMMARY
# ==========================================================

def print_suite_summary(
    case_results,
    aggregate
):

    print(
        "\n\n"
    )

    print_separator()

    print(
        "AUREON HELD-OUT BENCHMARK — FINAL SUMMARY"
    )

    print_separator()


    print(
        "\nDevelopment / pilot case excluded:"
    )

    print(
        f"  {DEVELOPMENT_ECR}"
    )


    print(
        "\nHeld-out cases:"
    )

    for ecr_id in BENCHMARK_ECRS:

        print(
            f"  {ecr_id}"
        )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "PER-ECR PERFORMANCE"
    )

    print(
        "-" * 90
    )


    header = (
        f"{'ECR':<10}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )


    print(
        "\n"
        + header
    )

    print(
        "-" * len(
            header
        )
    )


    for result in case_results:

        if result.get(
            "status"
        ) != "success":

            print(
                f"{result['ecr_id']:<10}"
                f"{'FAILED':>54}"
            )

            continue


        print(
            f"{result['ecr_id']:<10}"
            f"{result['tp']:>6}"
            f"{result['fp']:>6}"
            f"{result['fn']:>6}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
        )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "AGGREGATE PERFORMANCE"
    )

    print(
        "-" * 90
    )


    print(
        f"\nSuccessful held-out cases: "
        f"{aggregate['successful_cases']}"
    )

    print(
        f"Failed cases: "
        f"{aggregate['failed_cases']}"
    )


    if aggregate[
        "successful_cases"
    ] == 0:

        return


    print(
        "\nMACRO AVERAGES"
    )

    print(
        f"  Precision: "
        f"{aggregate['macro_precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{aggregate['macro_recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{aggregate['macro_f1']:.4f}"
    )


    print(
        "\nMICRO TOTALS"
    )

    print(
        f"  TP: "
        f"{aggregate['micro_tp']}"
    )

    print(
        f"  FP: "
        f"{aggregate['micro_fp']}"
    )

    print(
        f"  FN: "
        f"{aggregate['micro_fn']}"
    )


    print(
        "\nMICRO METRICS"
    )

    print(
        f"  Precision: "
        f"{aggregate['micro_precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{aggregate['micro_recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{aggregate['micro_f1']:.4f}"
    )


    print(
        "\nEFFICIENCY"
    )

    print(
        "  Mean tool calls: "
        f"{aggregate['mean_tool_calls']:.2f}"
        if aggregate[
            "mean_tool_calls"
        ] is not None
        else "  Mean tool calls: N/A"
    )

    print(
        "  Mean reasoning cycles: "
        f"{aggregate['mean_reasoning_cycles']:.2f}"
        if aggregate[
            "mean_reasoning_cycles"
        ] is not None
        else "  Mean reasoning cycles: N/A"
    )

    print(
        "  Mean total tokens: "
        f"{aggregate['mean_total_tokens']:.2f}"
        if aggregate[
            "mean_total_tokens"
        ] is not None
        else "  Mean total tokens: N/A"
    )

    print(
        "  Mean runtime: "
        f"{aggregate['mean_runtime_seconds']:.2f} s"
        if aggregate[
            "mean_runtime_seconds"
        ] is not None
        else "  Mean runtime: N/A"
    )


# ==========================================================
# MAIN BENCHMARK SUITE
# ==========================================================

if __name__ == "__main__":

    print(
        "\n"
    )

    print_separator()

    print(
        "AUREON AUTONOMOUS AI-PLM AGENT"
    )

    print(
        "Phase 5C.2 - Held-Out Benchmark Suite"
    )

    print_separator()


    print(
        f"\nModel: {MODEL}"
    )


    print(
        "\nDevelopment case excluded:"
    )

    print(
        f"  {DEVELOPMENT_ECR}"
    )


    print(
        "\nHeld-out evaluation cases:"
    )

    for ecr_id in BENCHMARK_ECRS:

        print(
            f"  - {ecr_id}"
        )


    # ======================================================
    # FREEZE AGENT
    # ======================================================

    frozen_agent_hash = (
        calculate_sha256(
            AGENT_FILE
        )
    )


    print(
        "\nFrozen agent SHA-256:"
    )

    print(
        frozen_agent_hash
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not edit agent/tool_calling_agent.py while "
        "this benchmark suite is running."
    )


    suite_started_at = (
        datetime.now().isoformat()
    )


    overall_start = (
        time.perf_counter()
    )


    case_results = []


    # ======================================================
    # RUN ALL HELD-OUT ECRs
    # ======================================================

    for index, ecr_id in enumerate(
        BENCHMARK_ECRS,
        start=1
    ):

        print(
            "\n\n"
        )

        print_separator(
            "#"
        )

        print(
            f"BENCHMARK {index}/{len(BENCHMARK_ECRS)}"
        )

        print(
            ecr_id
        )

        print_separator(
            "#"
        )


        try:

            result = run_single_case(
                ecr_id=ecr_id,
                frozen_agent_hash=
                    frozen_agent_hash
            )


        except Exception as error:

            print(
                "\nBENCHMARK CASE FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )


            result = {

                "ecr_id":
                    ecr_id,

                "status":
                    "error",

                "error_type":
                    type(
                        error
                    ).__name__,

                "error":
                    str(
                        error
                    )
            }


        case_results.append(
            result
        )


    # ======================================================
    # VERIFY AGENT STILL FROZEN
    # ======================================================

    final_agent_hash = (
        calculate_sha256(
            AGENT_FILE
        )
    )


    agent_unchanged = (
        final_agent_hash
        == frozen_agent_hash
    )


    # ======================================================
    # AGGREGATE
    # ======================================================

    aggregate = (
        calculate_aggregate_metrics(
            case_results
        )
    )


    overall_elapsed = round(
        time.perf_counter()
        - overall_start,
        3
    )


    suite_result = {

        "benchmark_name":
            (
                "Aureon Autonomous AI-PLM "
                "Held-Out Benchmark"
            ),

        "benchmark_phase":
            "5C.2",

        "development_case_excluded":
            DEVELOPMENT_ECR,

        "held_out_ecrs":
            BENCHMARK_ECRS,

        "model":
            MODEL,

        "suite_started_at":
            suite_started_at,

        "suite_completed_at":
            datetime.now().isoformat(),

        "agent_file":
            str(
                AGENT_FILE
            ),

        "frozen_agent_sha256":
            frozen_agent_hash,

        "final_agent_sha256":
            final_agent_hash,

        "agent_unchanged_during_suite":
            agent_unchanged,

        "total_suite_elapsed_seconds":
            overall_elapsed,

        "case_results":
            case_results,

        "aggregate_metrics":
            aggregate
    }


    # ======================================================
    # SAVE RESULTS
    # ======================================================

    suite_json = (
        save_suite_json(
            suite_result
        )
    )


    suite_csv = (
        save_suite_csv(
            case_results
        )
    )


    # ======================================================
    # DISPLAY FINAL SUMMARY
    # ======================================================

    print_suite_summary(
        case_results,
        aggregate
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "EXPERIMENTAL INTEGRITY"
    )

    print(
        "-" * 90
    )


    print(
        f"\nAgent unchanged during suite: "
        f"{agent_unchanged}"
    )


    print(
        "\nInitial SHA-256:"
    )

    print(
        frozen_agent_hash
    )


    print(
        "\nFinal SHA-256:"
    )

    print(
        final_agent_hash
    )


    print(
        f"\nTotal suite runtime: "
        f"{overall_elapsed:.2f} seconds"
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "BENCHMARK OUTPUT FILES"
    )

    print(
        "=" * 90
    )


    print(
        "\nFull benchmark JSON:"
    )

    print(
        suite_json
    )


    print(
        "\nPer-ECR benchmark CSV:"
    )

    print(
        suite_csv
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "HELD-OUT BENCHMARK SUITE COMPLETE"
    )

    print(
        "=" * 90
    )