# ==========================================================
# AUREON AUTONOMOUS AI-PLM AGENT
# Phase 5D.2
#
# D2.2 Architecture:
#
# Focused Autonomous Retrieval
#        ↓
# Gap-Directed Retrieval
#        ↓
# Evidence-Scored Candidate Assessment
#        ↓
# Deterministic Candidate Selection
#        ↓
# Final Machine-Readable Prediction
#
# D1, D2 and D2.1 remain unchanged.
# ==========================================================


# ==========================================================
# WINDOWS DLL INITIALIZATION
# ==========================================================

import torch


import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==========================================================
# IMPORT D1 INFRASTRUCTURE
# ==========================================================

from agent.tool_calling_agent import (
    client,
    MODEL,
    extract_prediction_json,
    validate_prediction,
    is_impact_entity
)


# ==========================================================
# IMPORT D2 INFRASTRUCTURE
# ==========================================================

from agent.tool_calling_agent_d2 import (
    run_retrieval_agent
)


# ==========================================================
# IMPORT D2.1 GAP-DIRECTED INFRASTRUCTURE
# ==========================================================

from agent.tool_calling_agent_d21 import (
    run_gap_planner,
    execute_gap_plan,
    INITIAL_TOOL_BUDGET
)


# ==========================================================
# D2.2 CONFIGURATION
# ==========================================================

MAX_GAP_TOOL_CALLS = 8

INCLUSION_THRESHOLD = 4


# ==========================================================
# OUTPUT FOLDER
# ==========================================================

D22_PREDICTIONS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "predictions_d22"
)

D22_PREDICTIONS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# EVIDENCE-SCORING RUBRIC
# ==========================================================

# Final score is computed programmatically.
#
# The LLM only identifies evidence dimensions.
#
# +3 Directly changed / invalidated
# +2 Explicit engineering relationship / PLM association
# +2 Supported engineering consequence
# +1 Requirement / validation support
# +1 Specific lifecycle action required
# -2 Context-only evidence
#
# Inclusion threshold = 4
#
# Direct-change entities are always included.


SCORING_RUBRIC = {

    "direct_change_evidence":
        3,

    "explicit_relationship":
        2,

    "engineering_consequence":
        2,

    "requirement_validation_support":
        1,

    "lifecycle_action":
        1,

    "context_only":
        -2
}


# ==========================================================
# CANDIDATE SCORER INSTRUCTIONS
# ==========================================================

CANDIDATE_SCORER_INSTRUCTIONS = """
You are the Evidence-Scored Candidate Assessment Layer of
the Aureon Autonomous AI-PLM Agent D2.2.

You receive:

1. the active engineering change;
2. the provisional engineering report;
3. the complete benchmark-safe evidence ledger;
4. any additional gap-directed evidence.

You have NO tools.

You have NO access to evaluator ground truth.

Your purpose is to identify plausible impacted entities and
describe the evidence dimensions supporting each candidate.

The application will calculate the numerical score.

You must NOT calculate or manipulate the final numerical
score yourself.

============================================================
IMPORTANT CONCEPT
============================================================

RELATED is not the same as IMPACTED.

RETRIEVED is not the same as IMPACTED.

EXISTS IN THE BOM is not automatically IMPACTED.

BELONGS TO THE PRODUCT is not automatically IMPACTED.

EXISTING SUPPLIER APPROVAL is not automatically IMPACTED.

A TEST existing near a changed subsystem is not
automatically IMPACTED.

============================================================
CANDIDATE POOL
============================================================

Include only entities that have a plausible engineering or
lifecycle impact path from the active change.

Every entity_id must appear somewhere in the supplied
evidence ledger.

Do not invent IDs.

Do not include IMPACT-* entities.

============================================================
EVIDENCE DIMENSIONS
============================================================

For every candidate, classify each of these as true or
false.

1. direct_change_evidence

TRUE only when the active engineering change directly
changes, invalidates, revises or explicitly targets this
entity.

Examples:
- primary component explicitly changed;
- requirement baseline explicitly conflicts with change.

Do not mark TRUE merely because the entity is related.

------------------------------------------------------------

2. explicit_relationship

TRUE when PLM evidence establishes a meaningful engineering
or lifecycle relationship connecting this entity to the
changed configuration.

Examples:
- manages
- cooled_by
- mounted_on
- connected_by
- verifies
- assembly operation for changed component
- service record explicitly concerning the affected item

Mere product membership should be treated cautiously.

------------------------------------------------------------

3. engineering_consequence

TRUE when the change produces a specific, intelligible
engineering reason why this entity may need modification,
compatibility checking, verification or disposition.

Examples:
- changed battery capacity may require BMS calibration /
  compatibility review;
- changed mass may affect mounting;
- changed thermal characteristics may affect cooling;
- changed electrical loading may affect interfaces.

Do not mark TRUE for generic relevance.

------------------------------------------------------------

4. requirement_validation_support

TRUE when:

- the entity is an affected requirement;
- the entity is a validation record that verifies an
  affected requirement/design characteristic;
- or explicit requirement/verification evidence strongly
  supports the impact path.

------------------------------------------------------------

5. lifecycle_action

TRUE only when evidence supports a specific PLM action such
as:

- revise;
- requalify;
- verify;
- retest;
- revalidate;
- update manufacturing operation;
- supplier reapproval;
- service action;
- configuration disposition.

A generic statement such as "review the product" is not
enough.

------------------------------------------------------------

6. context_only

TRUE when the entity was useful for understanding the
change but available evidence does not support treating the
entity itself as impacted or review-required.

Examples may include:

- general product record;
- generic document;
- unrelated historical record;
- supplier record with no sourcing consequence;
- BOM membership with no specific configuration consequence.

============================================================
SECOND-ORDER PROPAGATION
============================================================

Second-order entities are allowed.

But they require:

- an understandable engineering path;
- evidence in the supplied ledger;
- and a specific engineering consequence.

Do not exclude software, sensing, thermal or structural
entities merely because they are second-order.

Do not include them merely because they exist.

============================================================
REQUIREMENTS
============================================================

A requirement is a valid candidate when:

- its baseline directly conflicts with the change;
- the change may affect satisfaction of the requirement;
- or engineering verification is materially required.

Do not include broad subsystem requirements without a
specific impact path.

============================================================
VALIDATION
============================================================

A TEST-* entity is a valid candidate only if evidence shows
that it verifies a requirement/design characteristic
materially implicated by the change.

Baseline test success does not validate the changed
configuration.

============================================================
MANUFACTURING
============================================================

An MFG-* record may be a candidate when the changed
configuration could alter or require review/requalification
of that operation.

Do not include manufacturing solely because it manufactures
the same product.

============================================================
SERVICE
============================================================

An SRV-* entity may be a candidate when the change materially
affects replacement, diagnostics, serviceability,
maintenance, field action or service instructions.

============================================================
SUPPLIER
============================================================

A SUP-* entity requires evidence that sourcing,
qualification, approval or deliverables may change.

Existing approval by itself is context only.

============================================================
BOM / PRODUCT / DOCUMENT / SPECIFICATION
============================================================

Be conservative.

Include these only when evidence supports a specific
revision, effectivity, configuration or technical
disposition.

Do not include them merely because they exist.

============================================================
IMPACT LEVEL
============================================================

Assign one of:

Direct
First-order
Second-order
Lifecycle

This is descriptive only and does not determine the score.

============================================================
OUTPUT
============================================================

First provide a short explanation.

Then output exactly:

CANDIDATE_SCORE_JSON

{
  "ecr_id": "ECR-001",
  "candidates": [
    {
      "entity_id": "BAT-001",
      "impact_level": "Direct",
      "direct_change_evidence": true,
      "explicit_relationship": true,
      "engineering_consequence": true,
      "requirement_validation_support": false,
      "lifecycle_action": true,
      "context_only": false,
      "evidence_summary": "Short evidence-grounded summary."
    }
  ]
}

Rules:

- Every boolean must be true or false.
- Every candidate must contain all fields.
- Every entity ID must appear in supplied evidence.
- No IMPACT-* entities.
- No duplicate entity IDs.
- Do not include a numerical score.
- Do not place text after the JSON.
"""


# ==========================================================
# USAGE HELPER
# ==========================================================

def usage_dict(
    response
):

    result = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }


    usage = getattr(
        response,
        "usage",
        None
    )


    if usage is None:
        return result


    result[
        "input_tokens"
    ] = (
        getattr(
            usage,
            "input_tokens",
            0
        )
        or 0
    )


    result[
        "output_tokens"
    ] = (
        getattr(
            usage,
            "output_tokens",
            0
        )
        or 0
    )


    result[
        "total_tokens"
    ] = (
        getattr(
            usage,
            "total_tokens",
            0
        )
        or 0
    )


    return result


# ==========================================================
# GENERIC MARKER JSON EXTRACTION
# ==========================================================

def extract_marker_json(
    response_text,
    marker
):

    if not response_text:
        return None


    marker_index = response_text.rfind(
        marker
    )


    if marker_index == -1:
        return None


    remaining = response_text[
        marker_index
        + len(marker):
    ].strip()


    remaining = re.sub(
        r"^```json\s*",
        "",
        remaining,
        flags=re.IGNORECASE
    )


    first_brace = remaining.find(
        "{"
    )


    if first_brace == -1:
        return None


    decoder = json.JSONDecoder()


    try:

        value, _ = decoder.raw_decode(
            remaining[
                first_brace:
            ]
        )

        return value


    except json.JSONDecodeError:

        return None


# ==========================================================
# RUN CANDIDATE SCORER
# ==========================================================

def run_candidate_scorer(
    ecr_id,
    provisional_answer,
    evidence_ledger
):

    start_time = time.perf_counter()


    payload = {

        "active_ecr":
            ecr_id,

        "provisional_report":
            provisional_answer,

        "evidence_ledger":
            evidence_ledger
    }


    prompt = (
        "Assess evidence-supported impact candidates for "
        "the active engineering change.\n\n"
        "Do not use tools.\n"
        "Do not use ground truth.\n"
        "Do not calculate the numerical score.\n"
        "Evaluate the evidence dimensions exactly as "
        "defined in your instructions.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str
        )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 EVIDENCE-SCORED CANDIDATE ASSESSMENT"
    )

    print(
        "=" * 90
    )


    response = client.responses.create(
        model=MODEL,
        instructions=
            CANDIDATE_SCORER_INSTRUCTIONS,
        input=[
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ]
    )


    elapsed = round(
        time.perf_counter()
        - start_time,
        3
    )


    usage = usage_dict(
        response
    )


    candidate_data = (
        extract_marker_json(
            response.output_text,
            "CANDIDATE_SCORE_JSON"
        )
    )


    return {

        "answer":
            response.output_text,

        "candidate_data":
            candidate_data,

        "metrics": {

            "input_tokens":
                usage[
                    "input_tokens"
                ],

            "output_tokens":
                usage[
                    "output_tokens"
                ],

            "total_tokens":
                usage[
                    "total_tokens"
                ],

            "elapsed_seconds":
                elapsed
        }
    }


# ==========================================================
# CANDIDATE JSON VALIDATION
# ==========================================================

def validate_candidate_data(
    candidate_data,
    expected_ecr_id,
    evidence_ledger
):

    errors = []


    if candidate_data is None:

        errors.append(
            (
                "Candidate scoring JSON was not found "
                "or could not be parsed."
            )
        )

        return errors


    if not isinstance(
        candidate_data,
        dict
    ):

        errors.append(
            "Candidate scoring output must be an object."
        )

        return errors


    ecr_id = str(
        candidate_data.get(
            "ecr_id",
            ""
        )
    ).strip().upper()


    if ecr_id != str(
        expected_ecr_id
    ).strip().upper():

        errors.append(
            (
                "Candidate scoring ECR does not match "
                f"{expected_ecr_id}."
            )
        )


    candidates = candidate_data.get(
        "candidates"
    )


    if not isinstance(
        candidates,
        list
    ):

        errors.append(
            "candidates must be a list."
        )

        return errors


    evidence_text = json.dumps(
        evidence_ledger,
        ensure_ascii=False,
        default=str
    ).upper()


    allowed_levels = {
        "Direct",
        "First-order",
        "Second-order",
        "Lifecycle"
    }


    required_boolean_fields = [
        "direct_change_evidence",
        "explicit_relationship",
        "engineering_consequence",
        "requirement_validation_support",
        "lifecycle_action",
        "context_only"
    ]


    seen_entities = set()


    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        if not isinstance(
            candidate,
            dict
        ):

            errors.append(
                (
                    f"Candidate {index} "
                    "is not an object."
                )
            )

            continue


        entity_id = str(
            candidate.get(
                "entity_id",
                ""
            )
        ).strip().upper()


        if not entity_id:

            errors.append(
                (
                    f"Candidate {index} "
                    "has no entity_id."
                )
            )

            continue


        if entity_id in seen_entities:

            errors.append(
                (
                    "Duplicate candidate entity: "
                    f"{entity_id}"
                )
            )


        seen_entities.add(
            entity_id
        )


        if is_impact_entity(
            entity_id
        ):

            errors.append(
                (
                    f"{entity_id} is prohibited."
                )
            )


        if entity_id not in evidence_text:

            errors.append(
                (
                    f"{entity_id} does not appear in "
                    "the retrieved evidence ledger."
                )
            )


        impact_level = candidate.get(
            "impact_level"
        )


        if impact_level not in allowed_levels:

            errors.append(
                (
                    f"{entity_id} has invalid "
                    f"impact_level: {impact_level}"
                )
            )


        for field_name in (
            required_boolean_fields
        ):

            if not isinstance(
                candidate.get(
                    field_name
                ),
                bool
            ):

                errors.append(
                    (
                        f"{entity_id}: {field_name} "
                        "must be true or false."
                    )
                )


        evidence_summary = candidate.get(
            "evidence_summary"
        )


        if not isinstance(
            evidence_summary,
            str
        ) or not evidence_summary.strip():

            errors.append(
                (
                    f"{entity_id} requires a "
                    "non-empty evidence_summary."
                )
            )


    return errors


# ==========================================================
# CALCULATE EVIDENCE SCORE
# ==========================================================

def calculate_candidate_score(
    candidate
):

    score = 0


    if candidate.get(
        "direct_change_evidence",
        False
    ):

        score += SCORING_RUBRIC[
            "direct_change_evidence"
        ]


    if candidate.get(
        "explicit_relationship",
        False
    ):

        score += SCORING_RUBRIC[
            "explicit_relationship"
        ]


    if candidate.get(
        "engineering_consequence",
        False
    ):

        score += SCORING_RUBRIC[
            "engineering_consequence"
        ]


    if candidate.get(
        "requirement_validation_support",
        False
    ):

        score += SCORING_RUBRIC[
            "requirement_validation_support"
        ]


    if candidate.get(
        "lifecycle_action",
        False
    ):

        score += SCORING_RUBRIC[
            "lifecycle_action"
        ]


    if candidate.get(
        "context_only",
        False
    ):

        score += SCORING_RUBRIC[
            "context_only"
        ]


    return score


# ==========================================================
# DETERMINE FINAL CONFIDENCE
# ==========================================================

def determine_confidence(
    candidate,
    score
):

    if candidate.get(
        "direct_change_evidence",
        False
    ):

        return "Confirmed"


    if score >= 6:

        return "Strong candidate"


    return "Review candidate"


# ==========================================================
# DETERMINISTIC CANDIDATE SELECTION
# ==========================================================

def adjudicate_candidates(
    candidate_data
):
    """
    Convert evidence dimensions into the final prediction.

    Final selection is deterministic.
    """

    scored_candidates = []

    final_impacts = []


    for candidate in candidate_data.get(
        "candidates",
        []
    ):

        candidate = dict(
            candidate
        )


        score = calculate_candidate_score(
            candidate
        )


        candidate[
            "evidence_score"
        ] = score


        direct = candidate.get(
            "direct_change_evidence",
            False
        )


        context_only = candidate.get(
            "context_only",
            False
        )


        # --------------------------------------------------
        # DETERMINISTIC INCLUSION
        # --------------------------------------------------

        if direct:

            included = True

            decision_reason = (
                "Included because direct change evidence "
                "is present."
            )


        elif (
            score >= INCLUSION_THRESHOLD
            and not context_only
        ):

            included = True

            decision_reason = (
                f"Included because evidence score "
                f"{score} meets threshold "
                f"{INCLUSION_THRESHOLD}."
            )


        else:

            included = False

            if context_only:

                decision_reason = (
                    "Excluded because evidence indicates "
                    "context-only relevance."
                )

            else:

                decision_reason = (
                    f"Excluded because evidence score "
                    f"{score} is below threshold "
                    f"{INCLUSION_THRESHOLD}."
                )


        candidate[
            "included"
        ] = included


        candidate[
            "decision_reason"
        ] = decision_reason


        scored_candidates.append(
            candidate
        )


        if included:

            confidence = (
                determine_confidence(
                    candidate,
                    score
                )
            )


            final_impacts.append(
                {

                    "entity_id":
                        candidate[
                            "entity_id"
                        ],

                    "impact_level":
                        candidate[
                            "impact_level"
                        ],

                    "confidence":
                        confidence,

                    "reason":
                        candidate[
                            "evidence_summary"
                        ],

                    "evidence_score":
                        score
                }
            )


    final_prediction = {

        "ecr_id":
            candidate_data[
                "ecr_id"
            ],

        "predicted_impacts":
            final_impacts
    }


    return {
        "scored_candidates":
            scored_candidates,

        "final_prediction":
            final_prediction
    }


# ==========================================================
# FINAL PREDICTION VALIDATION
# ==========================================================

def validate_final_prediction(
    prediction,
    expected_ecr_id
):

    # validate_prediction() does not object to the extra
    # evidence_score field.

    errors = validate_prediction(
        prediction,
        expected_ecr_id
    )


    return errors


# ==========================================================
# SAVE D2.2 RESULT
# ==========================================================

def save_d22_result(
    ecr_id,
    benchmark_mode,
    provisional_prediction,
    gap_plan,
    scored_candidates,
    final_prediction,
    retrieval_metrics,
    planner_metrics,
    gap_execution_metrics,
    scorer_metrics,
    candidate_scorer_answer
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
        f"d22_"
        f"{mode_name}_"
        f"{timestamp}.json"
    )


    output_path = (
        D22_PREDICTIONS_FOLDER
        / filename
    )


    total_tokens = (
        retrieval_metrics.get(
            "total_tokens",
            0
        )
        + planner_metrics.get(
            "total_tokens",
            0
        )
        + scorer_metrics.get(
            "total_tokens",
            0
        )
    )


    total_tools = (
        retrieval_metrics.get(
            "tool_calls_executed",
            0
        )
        + gap_execution_metrics.get(
            "gap_calls_executed",
            0
        )
    )


    total_elapsed = round(
        retrieval_metrics.get(
            "elapsed_seconds",
            0
        )
        + planner_metrics.get(
            "elapsed_seconds",
            0
        )
        + gap_execution_metrics.get(
            "elapsed_seconds",
            0
        )
        + scorer_metrics.get(
            "elapsed_seconds",
            0
        ),
        3
    )


    payload = {

        "system":
            "Aureon Autonomous AI-PLM Agent",

        "architecture":
            (
                "D2.2 - Gap-Directed Retrieval + "
                "Evidence-Scored Candidate Adjudication"
            ),

        "phase":
            "5D.2",

        "model":
            MODEL,

        "ecr_id":
            ecr_id,

        "benchmark_mode":
            benchmark_mode,

        "created_at":
            datetime.now().isoformat(),

        "scoring_rubric":
            SCORING_RUBRIC,

        "inclusion_threshold":
            INCLUSION_THRESHOLD,

        "provisional_prediction":
            provisional_prediction,

        "gap_plan":
            gap_plan,

        "scored_candidates":
            scored_candidates,

        "prediction":
            final_prediction,

        "retrieval_metrics":
            retrieval_metrics,

        "planner_metrics":
            planner_metrics,

        "gap_execution_metrics":
            gap_execution_metrics,

        "scorer_metrics":
            scorer_metrics,

        "combined_metrics": {

            "tool_calls_executed":
                total_tools,

            "total_tokens":
                total_tokens,

            "elapsed_seconds":
                total_elapsed
        },

        "candidate_scorer_answer":
            candidate_scorer_answer
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
# DISPLAY SCORE TABLE
# ==========================================================

def print_candidate_scores(
    scored_candidates
):

    print(
        "\n"
        + "=" * 115
    )

    print(
        "D2.2 CANDIDATE EVIDENCE SCORECARD"
    )

    print(
        "=" * 115
    )


    header = (
        f"{'Entity':<14}"
        f"{'Dir':>5}"
        f"{'Rel':>5}"
        f"{'Eng':>5}"
        f"{'Req/V':>7}"
        f"{'Life':>6}"
        f"{'Ctx':>5}"
        f"{'Score':>7}"
        f"{'Decision':>12}"
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


    for candidate in scored_candidates:

        print(
            f"{candidate['entity_id']:<14}"
            f"{str(candidate['direct_change_evidence'])[0]:>5}"
            f"{str(candidate['explicit_relationship'])[0]:>5}"
            f"{str(candidate['engineering_consequence'])[0]:>5}"
            f"{str(candidate['requirement_validation_support'])[0]:>7}"
            f"{str(candidate['lifecycle_action'])[0]:>6}"
            f"{str(candidate['context_only'])[0]:>5}"
            f"{candidate['evidence_score']:>7}"
            f"{('INCLUDE' if candidate['included'] else 'EXCLUDE'):>12}"
        )


# ==========================================================
# COMPLETE D2.2 PIPELINE
# ==========================================================

def analyse_change_d22(
    ecr_id,
    benchmark_mode=False
):

    overall_start = time.perf_counter()


    ecr_id = (
        str(ecr_id)
        .strip()
        .upper()
    )


    # ======================================================
    # STAGE 1
    # FOCUSED INITIAL RETRIEVAL
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 STAGE 1 - FOCUSED AUTONOMOUS RETRIEVAL"
    )

    print(
        "=" * 90
    )


    retrieval_result = (
        run_retrieval_agent(
            ecr_id=ecr_id,
            benchmark_mode=benchmark_mode,
            max_tool_calls=
                INITIAL_TOOL_BUDGET
        )
    )


    provisional_answer = (
        retrieval_result[
            "answer"
        ]
    )


    provisional_prediction = (
        extract_prediction_json(
            provisional_answer
        )
    )


    provisional_errors = (
        validate_prediction(
            provisional_prediction,
            expected_ecr_id=ecr_id
        )
    )


    if provisional_errors:

        return {

            "status":
                "provisional_prediction_failed",

            "errors":
                provisional_errors
        }


    # ======================================================
    # STAGE 2
    # GAP PLANNER
    # ======================================================

    gap_planner_result = (
        run_gap_planner(
            ecr_id=ecr_id,
            provisional_answer=
                provisional_answer,
            evidence_ledger=
                retrieval_result[
                    "evidence_ledger"
                ]
        )
    )


    if gap_planner_result[
        "errors"
    ]:

        return {

            "status":
                "gap_plan_failed",

            "errors":
                gap_planner_result[
                    "errors"
                ]
        }


    gap_plan = (
        gap_planner_result[
            "gap_plan"
        ]
    )


    # ======================================================
    # STAGE 3
    # TARGETED GAP RETRIEVAL
    # ======================================================

    gap_execution = (
        execute_gap_plan(
            ecr_id=ecr_id,
            gap_plan=gap_plan,
            evidence_ledger=
                retrieval_result[
                    "evidence_ledger"
                ],
            benchmark_mode=
                benchmark_mode
        )
    )


    expanded_ledger = (
        gap_execution[
            "expanded_evidence_ledger"
        ]
    )


    # ======================================================
    # STAGE 4
    # EVIDENCE-SCORED CANDIDATE GENERATION
    # ======================================================

    scorer_result = (
        run_candidate_scorer(
            ecr_id=ecr_id,
            provisional_answer=
                provisional_answer,
            evidence_ledger=
                expanded_ledger
        )
    )


    candidate_data = (
        scorer_result[
            "candidate_data"
        ]
    )


    candidate_errors = (
        validate_candidate_data(
            candidate_data=
                candidate_data,
            expected_ecr_id=
                ecr_id,
            evidence_ledger=
                expanded_ledger
        )
    )


    if candidate_errors:

        return {

            "status":
                "candidate_scoring_failed",

            "errors":
                candidate_errors,

            "candidate_scorer_answer":
                scorer_result[
                    "answer"
                ]
        }


    # ======================================================
    # STAGE 5
    # DETERMINISTIC ADJUDICATION
    # ======================================================

    adjudication = (
        adjudicate_candidates(
            candidate_data
        )
    )


    scored_candidates = (
        adjudication[
            "scored_candidates"
        ]
    )


    final_prediction = (
        adjudication[
            "final_prediction"
        ]
    )


    final_errors = (
        validate_final_prediction(
            prediction=
                final_prediction,
            expected_ecr_id=
                ecr_id
        )
    )


    if final_errors:

        return {

            "status":
                "final_prediction_failed",

            "errors":
                final_errors
        }


    # ======================================================
    # SAVE
    # ======================================================

    output_file = (
        save_d22_result(
            ecr_id=ecr_id,
            benchmark_mode=
                benchmark_mode,
            provisional_prediction=
                provisional_prediction,
            gap_plan=
                gap_plan,
            scored_candidates=
                scored_candidates,
            final_prediction=
                final_prediction,
            retrieval_metrics=
                retrieval_result[
                    "metrics"
                ],
            planner_metrics=
                gap_planner_result[
                    "metrics"
                ],
            gap_execution_metrics=
                gap_execution[
                    "metrics"
                ],
            scorer_metrics=
                scorer_result[
                    "metrics"
                ],
            candidate_scorer_answer=
                scorer_result[
                    "answer"
                ]
        )
    )


    overall_elapsed = round(
        time.perf_counter()
        - overall_start,
        3
    )


    return {

        "status":
            "success",

        "ecr_id":
            ecr_id,

        "benchmark_mode":
            benchmark_mode,

        "provisional_prediction":
            provisional_prediction,

        "gap_plan":
            gap_plan,

        "gap_execution_log":
            gap_execution[
                "execution_log"
            ],

        "scored_candidates":
            scored_candidates,

        "final_prediction":
            final_prediction,

        "retrieval_metrics":
            retrieval_result[
                "metrics"
            ],

        "planner_metrics":
            gap_planner_result[
                "metrics"
            ],

        "gap_execution_metrics":
            gap_execution[
                "metrics"
            ],

        "scorer_metrics":
            scorer_result[
                "metrics"
            ],

        "overall_elapsed_seconds":
            overall_elapsed,

        "prediction_file":
            output_file
    }


# ==========================================================
# COMMAND LINE
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
        "Phase 5D.2 - Evidence-Scored Candidate Adjudication"
    )

    print(
        "=" * 90
    )


    print(
        f"\nModel: {MODEL}"
    )


    print(
        "\nArchitecture:"
    )

    print(
        "  Stage 1 - Focused autonomous retrieval"
    )

    print(
        "  Stage 2 - Gap-directed retrieval planner"
    )

    print(
        "  Stage 3 - Targeted PLM evidence acquisition"
    )

    print(
        "  Stage 4 - Candidate evidence classification"
    )

    print(
        "  Stage 5 - Deterministic scoring and selection"
    )


    print(
        "\nEvidence scoring rubric:"
    )

    print(
        "  Direct change evidence        +3"
    )

    print(
        "  Explicit relationship         +2"
    )

    print(
        "  Engineering consequence       +2"
    )

    print(
        "  Requirement/validation support +1"
    )

    print(
        "  Lifecycle action              +1"
    )

    print(
        "  Context-only                  -2"
    )


    print(
        f"\nInclusion threshold: "
        f"{INCLUSION_THRESHOLD}"
    )


    ecr_id = input(
        "\nEnter ECR ID "
        "(development case ECR-001): "
    ).strip().upper()


    print(
        "\nAnalysis mode:"
    )

    print(
        "1 - Normal"
    )

    print(
        "2 - Benchmark-safe"
    )


    mode = input(
        "\nSelect mode [2]: "
    ).strip()


    if not mode:
        mode = "2"


    benchmark_mode = (
        mode == "2"
    )


    print(
        "\nFor Phase 5D.2 use ECR-001 "
        "as the development case."
    )


    result = analyse_change_d22(
        ecr_id=ecr_id,
        benchmark_mode=benchmark_mode
    )


    # ======================================================
    # FAILURE
    # ======================================================

    if result[
        "status"
    ] != "success":

        print(
            "\n"
            + "=" * 90
        )

        print(
            "D2.2 ANALYSIS FAILED"
        )

        print(
            "=" * 90
        )


        for error in result.get(
            "errors",
            []
        ):

            print(
                f"- {error}"
            )


        if result.get(
            "candidate_scorer_answer"
        ):

            print(
                "\nCandidate scorer response:"
            )

            print(
                result[
                    "candidate_scorer_answer"
                ]
            )


        sys.exit(
            1
        )


    # ======================================================
    # PROVISIONAL PREDICTION
    # ======================================================

    provisional_entities = [
        item[
            "entity_id"
        ]
        for item in result[
            "provisional_prediction"
        ][
            "predicted_impacts"
        ]
    ]


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 PROVISIONAL PREDICTION"
    )

    print(
        "=" * 90
    )


    print(
        f"\nProvisional entities: "
        f"{len(provisional_entities)}"
    )


    for entity_id in provisional_entities:

        print(
            f"  - {entity_id}"
        )


    # ======================================================
    # GAP PLAN
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 GAP PLAN"
    )

    print(
        "=" * 90
    )


    gaps = result[
        "gap_plan"
    ].get(
        "gaps",
        []
    )


    print(
        f"\nTargeted calls planned: "
        f"{len(gaps)}"
    )


    for index, gap in enumerate(
        gaps,
        start=1
    ):

        print(
            f"\n{index}. "
            f"{gap.get('gap_category')}"
        )

        print(
            f"   Tool: "
            f"{gap.get('tool_name')}"
        )

        print(
            f"   Arguments: "
            f"{gap.get('arguments')}"
        )

        print(
            f"   Reason: "
            f"{gap.get('reason')}"
        )


    # ======================================================
    # SCORECARD
    # ======================================================

    print_candidate_scores(
        result[
            "scored_candidates"
        ]
    )


    # ======================================================
    # FINAL PREDICTION
    # ======================================================

    final_impacts = result[
        "final_prediction"
    ][
        "predicted_impacts"
    ]


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 FINAL PREDICTION"
    )

    print(
        "=" * 90
    )


    print(
        "\nPrediction validation PASSED."
    )


    print(
        f"\nFinal predicted entities: "
        f"{len(final_impacts)}"
    )


    for impact in final_impacts:

        print(
            f"  - "
            f"{impact['entity_id']} "
            f"| Score = "
            f"{impact['evidence_score']} "
            f"| "
            f"{impact['confidence']}"
        )


    # ======================================================
    # EXCLUDED CANDIDATES
    # ======================================================

    excluded = [
        item
        for item in result[
            "scored_candidates"
        ]
        if not item[
            "included"
        ]
    ]


    print(
        "\nExcluded candidates:"
    )


    if not excluded:

        print(
            "  None"
        )


    else:

        for item in excluded:

            print(
                f"  - "
                f"{item['entity_id']} "
                f"| Score = "
                f"{item['evidence_score']} "
                f"| "
                f"{item['decision_reason']}"
            )


    # ======================================================
    # METRICS
    # ======================================================

    retrieval_metrics = result[
        "retrieval_metrics"
    ]

    planner_metrics = result[
        "planner_metrics"
    ]

    gap_metrics = result[
        "gap_execution_metrics"
    ]

    scorer_metrics = result[
        "scorer_metrics"
    ]


    total_tools = (
        retrieval_metrics.get(
            "tool_calls_executed",
            0
        )
        + gap_metrics.get(
            "gap_calls_executed",
            0
        )
    )


    total_tokens = (
        retrieval_metrics.get(
            "total_tokens",
            0
        )
        + planner_metrics.get(
            "total_tokens",
            0
        )
        + scorer_metrics.get(
            "total_tokens",
            0
        )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.2 EXECUTION METRICS"
    )

    print(
        "=" * 90
    )


    print(
        f"\nInitial retrieval tools: "
        f"{retrieval_metrics.get('tool_calls_executed')}"
    )

    print(
        f"Gap-directed tools: "
        f"{gap_metrics.get('gap_calls_executed')}"
    )

    print(
        f"Total PLM tool calls: "
        f"{total_tools}"
    )


    print(
        f"\nRetrieval tokens: "
        f"{retrieval_metrics.get('total_tokens')}"
    )

    print(
        f"Planner tokens: "
        f"{planner_metrics.get('total_tokens')}"
    )

    print(
        f"Candidate scorer tokens: "
        f"{scorer_metrics.get('total_tokens')}"
    )

    print(
        f"Total model tokens: "
        f"{total_tokens}"
    )


    print(
        f"\nOverall elapsed time: "
        f"{result['overall_elapsed_seconds']} s"
    )


    print(
        "\nPrediction saved to:"
    )

    print(
        result[
            "prediction_file"
        ]
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "PHASE 5D.2 COMPLETE"
    )

    print(
        "=" * 90
    )