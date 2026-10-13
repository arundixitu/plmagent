import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

ANALYSIS_FOLDER = (
    RESULTS_FOLDER
    / "error_analysis"
)

ANALYSIS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# ENTITY CATEGORY CLASSIFICATION
# ==========================================================

def classify_entity(
    entity_id
):
    """
    Convert PLM entity IDs into lifecycle categories.
    """

    entity_id = str(
        entity_id
    ).strip().upper()


    if entity_id.startswith("REQ-"):
        return "Requirement"

    if entity_id.startswith("TEST-"):
        return "Validation"

    if entity_id.startswith("MFG-"):
        return "Manufacturing"

    if entity_id.startswith("SRV-"):
        return "Service"

    if entity_id.startswith("SUP-"):
        return "Supplier"

    if entity_id.startswith("BOM-"):
        return "BOM / Configuration"

    if entity_id.startswith("DOC-"):
        return "Document"

    if entity_id.startswith("SPEC-"):
        return "Design Specification"

    if entity_id.startswith("PROD-"):
        return "Product"

    if entity_id.startswith("FIELD-"):
        return "Field Failure"

    if entity_id.startswith("ECR-"):
        return "Engineering Change"

    if entity_id.startswith("IMPACT-"):
        return "Impact Record"


    # ------------------------------------------------------
    # KNOWN COMPONENT PREFIXES IN AUREON WORLD
    # ------------------------------------------------------

    component_prefixes = {

        "BAT",
        "BMS",
        "ENC",
        "MTR",
        "INV",
        "CHR",
        "CON",
        "CAB",
        "THM",
        "FAN",
        "FRM",
        "BRK",
        "WHL",
        "SUS",
        "DASH",
        "TCU",
        "CHG",
        "SEA",
        "LGT",
        "SNS",
        "SW"
    }


    prefix = entity_id.split(
        "-"
    )[0]


    if prefix in component_prefixes:
        return "Component"


    return "Other"


# ==========================================================
# FIND LATEST BENCHMARK SUITE
# ==========================================================

def find_latest_suite():
    """
    Find newest held-out benchmark suite JSON.
    """

    files = list(
        RESULTS_FOLDER.glob(
            "heldout_benchmark_suite_*.json"
        )
    )


    if not files:

        raise FileNotFoundError(
            (
                "No held-out benchmark suite JSON found in:\n"
                f"{RESULTS_FOLDER}"
            )
        )


    latest_file = max(
        files,
        key=lambda path:
            path.stat().st_mtime
    )


    return latest_file


# ==========================================================
# LOAD SUITE
# ==========================================================

def load_suite(
    suite_file
):

    with open(
        suite_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ==========================================================
# SAFE METRICS
# ==========================================================

def calculate_metrics(
    tp,
    fp,
    fn
):

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (
            tp + fp
        )


    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (
            tp + fn
        )


    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (
                precision + recall
            )
        )


    return {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    }


# ==========================================================
# ANALYSE BENCHMARK
# ==========================================================

def analyse_suite(
    suite
):
    """
    Perform error analysis across all successful held-out ECRs.
    """

    case_results = suite.get(
        "case_results",
        []
    )


    successful_cases = [
        case
        for case in case_results
        if case.get(
            "status"
        ) == "success"
    ]


    category_counts = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )


    missed_entity_counter = Counter()

    extra_entity_counter = Counter()

    correct_entity_counter = Counter()


    per_ecr_errors = []


    # ======================================================
    # PROCESS EACH ECR
    # ======================================================

    for case in successful_cases:

        ecr_id = case.get(
            "ecr_id"
        )


        true_positives = case.get(
            "true_positive_entities",
            []
        )


        false_positives = case.get(
            "false_positive_entities",
            []
        )


        false_negatives = case.get(
            "false_negative_entities",
            []
        )


        # --------------------------------------------------
        # TRUE POSITIVES
        # --------------------------------------------------

        for entity in true_positives:

            category = classify_entity(
                entity
            )

            category_counts[
                category
            ][
                "tp"
            ] += 1

            correct_entity_counter[
                entity
            ] += 1


        # --------------------------------------------------
        # FALSE POSITIVES
        # --------------------------------------------------

        for entity in false_positives:

            category = classify_entity(
                entity
            )

            category_counts[
                category
            ][
                "fp"
            ] += 1

            extra_entity_counter[
                entity
            ] += 1


        # --------------------------------------------------
        # FALSE NEGATIVES
        # --------------------------------------------------

        for entity in false_negatives:

            category = classify_entity(
                entity
            )

            category_counts[
                category
            ][
                "fn"
            ] += 1

            missed_entity_counter[
                entity
            ] += 1


        per_ecr_errors.append(
            {

                "ecr_id":
                    ecr_id,

                "tp":
                    len(
                        true_positives
                    ),

                "fp":
                    len(
                        false_positives
                    ),

                "fn":
                    len(
                        false_negatives
                    ),

                "false_positive_entities":
                    false_positives,

                "false_negative_entities":
                    false_negatives
            }
        )


    # ======================================================
    # CATEGORY PERFORMANCE
    # ======================================================

    category_results = []


    for category in sorted(
        category_counts.keys()
    ):

        counts = category_counts[
            category
        ]


        metrics = calculate_metrics(
            tp=counts["tp"],
            fp=counts["fp"],
            fn=counts["fn"]
        )


        gt_total = (
            counts["tp"]
            + counts["fn"]
        )


        predicted_total = (
            counts["tp"]
            + counts["fp"]
        )


        category_results.append(
            {

                "category":
                    category,

                "tp":
                    counts["tp"],

                "fp":
                    counts["fp"],

                "fn":
                    counts["fn"],

                "ground_truth_total":
                    gt_total,

                "predicted_total":
                    predicted_total,

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
                    ]
            }
        )


    # ======================================================
    # ERROR CATEGORY COUNTS
    # ======================================================

    missed_categories = Counter()

    extra_categories = Counter()


    for entity, count in (
        missed_entity_counter.items()
    ):

        missed_categories[
            classify_entity(
                entity
            )
        ] += count


    for entity, count in (
        extra_entity_counter.items()
    ):

        extra_categories[
            classify_entity(
                entity
            )
        ] += count


    # ======================================================
    # RESULT
    # ======================================================

    return {

        "successful_cases":
            len(
                successful_cases
            ),

        "category_results":
            category_results,

        "missed_categories":
            dict(
                missed_categories
            ),

        "extra_categories":
            dict(
                extra_categories
            ),

        "most_frequently_missed_entities":
            missed_entity_counter.most_common(),

        "most_frequent_extra_entities":
            extra_entity_counter.most_common(),

        "most_frequent_correct_entities":
            correct_entity_counter.most_common(),

        "per_ecr_errors":
            per_ecr_errors
    }


# ==========================================================
# PRINT CATEGORY TABLE
# ==========================================================

def print_category_table(
    category_results
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LIFECYCLE CATEGORY PERFORMANCE"
    )

    print(
        "=" * 100
    )


    header = (
        f"{'Category':<24}"
        f"{'GT':>6}"
        f"{'Pred':>7}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
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


    for row in category_results:

        print(
            f"{row['category']:<24}"
            f"{row['ground_truth_total']:>6}"
            f"{row['predicted_total']:>7}"
            f"{row['tp']:>6}"
            f"{row['fp']:>6}"
            f"{row['fn']:>6}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
        )


# ==========================================================
# PRINT RANKED COUNTERS
# ==========================================================

def print_ranked_counter(
    title,
    items,
    limit=20
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )


    if not items:

        print(
            "\nNone"
        )

        return


    for rank, (
        entity,
        count
    ) in enumerate(
        items[:limit],
        start=1
    ):

        category = classify_entity(
            entity
        )

        print(
            f"{rank:>2}. "
            f"{entity:<14} "
            f"{category:<22} "
            f"Count = {count}"
        )


# ==========================================================
# PRINT CATEGORY ERROR COUNTS
# ==========================================================

def print_category_errors(
    title,
    category_counts
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )


    sorted_items = sorted(
        category_counts.items(),
        key=lambda item:
            item[1],
        reverse=True
    )


    total = sum(
        category_counts.values()
    )


    for category, count in sorted_items:

        if total == 0:
            percentage = 0.0
        else:
            percentage = (
                100
                * count
                / total
            )


        print(
            f"{category:<25}"
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )


# ==========================================================
# SAVE CATEGORY CSV
# ==========================================================

def save_category_csv(
    category_results
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output_path = (
        ANALYSIS_FOLDER
        / (
            "category_performance_"
            f"{timestamp}.csv"
        )
    )


    columns = [
        "Category",
        "Ground_Truth_Total",
        "Predicted_Total",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1"
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


        for row in category_results:

            writer.writerow(
                {

                    "Category":
                        row[
                            "category"
                        ],

                    "Ground_Truth_Total":
                        row[
                            "ground_truth_total"
                        ],

                    "Predicted_Total":
                        row[
                            "predicted_total"
                        ],

                    "TP":
                        row[
                            "tp"
                        ],

                    "FP":
                        row[
                            "fp"
                        ],

                    "FN":
                        row[
                            "fn"
                        ],

                    "Precision":
                        round(
                            row[
                                "precision"
                            ],
                            6
                        ),

                    "Recall":
                        round(
                            row[
                                "recall"
                            ],
                            6
                        ),

                    "F1":
                        round(
                            row[
                                "f1"
                            ],
                            6
                        )
                }
            )


    return output_path


# ==========================================================
# SAVE ENTITY ERROR CSV
# ==========================================================

def save_entity_error_csv(
    analysis
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output_path = (
        ANALYSIS_FOLDER
        / (
            "entity_error_frequency_"
            f"{timestamp}.csv"
        )
    )


    missed = dict(
        analysis[
            "most_frequently_missed_entities"
        ]
    )


    extra = dict(
        analysis[
            "most_frequent_extra_entities"
        ]
    )


    correct = dict(
        analysis[
            "most_frequent_correct_entities"
        ]
    )


    all_entities = sorted(
        set(
            missed.keys()
        )
        | set(
            extra.keys()
        )
        | set(
            correct.keys()
        )
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Entity_ID",
                "Category",
                "Correct_Count",
                "Missed_Count",
                "Extra_Count"
            ]
        )


        writer.writeheader()


        for entity in all_entities:

            writer.writerow(
                {

                    "Entity_ID":
                        entity,

                    "Category":
                        classify_entity(
                            entity
                        ),

                    "Correct_Count":
                        correct.get(
                            entity,
                            0
                        ),

                    "Missed_Count":
                        missed.get(
                            entity,
                            0
                        ),

                    "Extra_Count":
                        extra.get(
                            entity,
                            0
                        )
                }
            )


    return output_path


# ==========================================================
# SAVE COMPLETE ANALYSIS JSON
# ==========================================================

def save_analysis_json(
    analysis,
    source_suite
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output_path = (
        ANALYSIS_FOLDER
        / (
            "benchmark_error_analysis_"
            f"{timestamp}.json"
        )
    )


    payload = {

        "analysis_name":
            (
                "Aureon Held-Out Benchmark "
                "Error Pattern Analysis"
            ),

        "analysis_phase":
            "5C.3",

        "created_at":
            datetime.now().isoformat(),

        "source_benchmark":
            str(
                source_suite
            ),

        "analysis":
            analysis
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
# IDENTIFY WEAKEST CATEGORIES
# ==========================================================

def print_weakest_categories(
    category_results
):

    eligible = [
        row
        for row in category_results
        if row[
            "ground_truth_total"
        ] > 0
    ]


    ranked = sorted(
        eligible,
        key=lambda row:
            (
                row[
                    "recall"
                ],
                row[
                    "f1"
                ]
            )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "LOWEST-RECALL LIFECYCLE CATEGORIES"
    )

    print(
        "=" * 90
    )


    for rank, row in enumerate(
        ranked,
        start=1
    ):

        print(
            f"{rank:>2}. "
            f"{row['category']:<24} "
            f"Recall = {row['recall']:.4f}   "
            f"F1 = {row['f1']:.4f}   "
            f"FN = {row['fn']}"
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUREON AUTONOMOUS AI-PLM AGENT"
    )

    print(
        "Phase 5C.3 - Benchmark Error-Pattern Analysis"
    )

    print(
        "=" * 100
    )


    try:

        # --------------------------------------------------
        # FIND BENCHMARK
        # --------------------------------------------------

        suite_file = (
            find_latest_suite()
        )


        print(
            "\nBenchmark suite:"
        )

        print(
            suite_file
        )


        # --------------------------------------------------
        # LOAD
        # --------------------------------------------------

        suite = load_suite(
            suite_file
        )


        # --------------------------------------------------
        # ANALYSE
        # --------------------------------------------------

        analysis = analyse_suite(
            suite
        )


        print(
            f"\nSuccessful held-out cases analysed: "
            f"{analysis['successful_cases']}"
        )


        # --------------------------------------------------
        # CATEGORY PERFORMANCE
        # --------------------------------------------------

        print_category_table(
            analysis[
                "category_results"
            ]
        )


        # --------------------------------------------------
        # ERROR DISTRIBUTION
        # --------------------------------------------------

        print_category_errors(
            "FALSE NEGATIVES BY LIFECYCLE CATEGORY",
            analysis[
                "missed_categories"
            ]
        )


        print_category_errors(
            "FALSE POSITIVES BY LIFECYCLE CATEGORY",
            analysis[
                "extra_categories"
            ]
        )


        # --------------------------------------------------
        # ENTITY FREQUENCY
        # --------------------------------------------------

        print_ranked_counter(
            "MOST FREQUENTLY MISSED ENTITIES",
            analysis[
                "most_frequently_missed_entities"
            ]
        )


        print_ranked_counter(
            "MOST FREQUENT EXTRA / OVER-PREDICTED ENTITIES",
            analysis[
                "most_frequent_extra_entities"
            ]
        )


        # --------------------------------------------------
        # WEAKEST CATEGORIES
        # --------------------------------------------------

        print_weakest_categories(
            analysis[
                "category_results"
            ]
        )


        # --------------------------------------------------
        # SAVE OUTPUT
        # --------------------------------------------------

        category_csv = (
            save_category_csv(
                analysis[
                    "category_results"
                ]
            )
        )


        entity_csv = (
            save_entity_error_csv(
                analysis
            )
        )


        analysis_json = (
            save_analysis_json(
                analysis,
                source_suite=suite_file
            )
        )


        print(
            "\n"
            + "=" * 100
        )

        print(
            "ERROR ANALYSIS SAVED"
        )

        print(
            "=" * 100
        )


        print(
            "\nCategory performance CSV:"
        )

        print(
            category_csv
        )


        print(
            "\nEntity error-frequency CSV:"
        )

        print(
            entity_csv
        )


        print(
            "\nComplete error-analysis JSON:"
        )

        print(
            analysis_json
        )


        print(
            "\n"
            + "=" * 100
        )

        print(
            "PHASE 5C.3 COMPLETE"
        )

        print(
            "=" * 100
        )


    except Exception as error:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "ERROR ANALYSIS FAILED"
        )

        print(
            "=" * 100
        )


        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )