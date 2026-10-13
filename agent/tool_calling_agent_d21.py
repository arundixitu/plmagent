# ==========================================================
# AUREON AUTONOMOUS AI-PLM AGENT
# Phase 5D.1
#
# D2.1 Architecture:
#
# Focused Autonomous Retrieval
#        ↓
# Provisional Prediction
#        ↓
# Gap-Directed Retrieval Planner
#        ↓
# Targeted Evidence Acquisition
#        ↓
# Independent Impact Adjudication
#        ↓
# Final Prediction
#
# D1 and D2 remain unchanged.
# ==========================================================


# ==========================================================
# WINDOWS DLL INITIALIZATION
# ==========================================================

# Keep torch first because of the Windows DLL behaviour
# already observed in this project.
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
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORT D1 INFRASTRUCTURE
# ==========================================================

from agent.tool_calling_agent import (
    client,
    MODEL,
    execute_tool,
    clean_json_value,
    sanitize_benchmark_result,
    result_contains_benchmark_withholding,
    extract_prediction_json,
    validate_prediction,
    build_tool_cache_key,
    is_impact_entity
)


# ==========================================================
# IMPORT D2 COMPONENTS
# ==========================================================

from agent.tool_calling_agent_d2 import (
    run_retrieval_agent,
    run_impact_adjudicator,
    validate_d2_prediction
)


# ==========================================================
# D2.1 CONFIGURATION
# ==========================================================

# Stage 1 should gather enough evidence to establish the
# change and its first-order neighbourhood, but it should
# not exhaust the complete budget.

INITIAL_TOOL_BUDGET = 18

MAX_GAP_TOOL_CALLS = 8

MAX_TOTAL_TOOL_CALLS = (
    INITIAL_TOOL_BUDGET
    + MAX_GAP_TOOL_CALLS
)


# ==========================================================
# OUTPUT FOLDER
# ==========================================================

D21_PREDICTIONS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "predictions_d21"
)

D21_PREDICTIONS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# GAP PLANNER INSTRUCTIONS
# ==========================================================

GAP_PLANNER_INSTRUCTIONS = """
You are the Lifecycle Gap-Directed Retrieval Planner for
the Aureon Autonomous AI-PLM Agent D2.1.

The first autonomous retrieval stage has already analysed an
engineering change and collected a partial evidence ledger.

Your task is NOT to make the final impact prediction.

Your task is to inspect the evidence already collected and
identify a small number of high-value evidence gaps that
should be investigated before final adjudication.

You have NO access to ground truth.

You have NO tools yourself.

You must propose targeted tool calls that the application
may execute.

============================================================
PRIMARY OBJECTIVE
============================================================

Identify where the current evidence may be insufficient to
support reliable engineering-change impact assessment.

Focus especially on meaningful second-order and lifecycle
propagation.

Do NOT request additional evidence merely for completeness.

Every proposed call should have a clear engineering reason.

============================================================
LIFECYCLE GAP CHECK
============================================================

Inspect whether sufficiently relevant evidence exists for:

1. Primary changed component
2. Directly connected components
3. Mechanical support / mounting
4. Electrical / power path
5. Thermal path
6. Software / control
7. Requirements
8. Validation / verification
9. Manufacturing
10. Supplier
11. Service / field
12. Configuration / documentation

Not every change affects every category.

Do not force calls into every category.

============================================================
PROPAGATION PRINCIPLE
============================================================

Look for evidence paths such as:

changed component
→ directly related subsystem
→ dependent component
→ requirement
→ validation
→ manufacturing / service consequence

However, stop where propagation becomes speculative.

============================================================
IMPORTANT FAILURE MODES TO AVOID
============================================================

Do NOT:

- request a component merely because it belongs to the same
  product;
- request every available requirement;
- request every nearby validation test;
- request every supplier or manufacturing record;
- request generic documents unless a specific evidence gap
  justifies them;
- request an entity solely to increase prediction count.

============================================================
HIGH-VALUE GAP EXAMPLES
============================================================

Examples of useful reasoning patterns:

THERMAL:
If a changed battery affects a cooling plate, ask whether
the cooling subsystem has dependent devices or controls that
may also require engineering review.

SOFTWARE / CONTROL:
If a changed component is managed by an electronic
controller, ask whether software, sensing, calibration or
control requirements need investigation.

VALIDATION:
If a requirement may be affected, check whether there are
specific tests associated with that requirement.

MECHANICAL:
If mass, dimensions or mounting may change, inspect
structural or mounting dependencies.

ELECTRICAL:
If current, voltage, power or interfaces may change, inspect
the actual electrical path and relevant requirements.

SERVICE:
Only investigate service records if the change may alter
replacement, diagnosis, serviceability, field action or
maintenance behaviour.

============================================================
ALLOWED TOOL NAMES
============================================================

You may propose ONLY these tool names:

get_component
get_related_components
get_design_specs
get_requirement
search_requirements
get_bom_for_component
get_supplier_for_component
get_manufacturing_for_component
get_validation_test
get_validation_for_requirement
get_service_for_component
get_field_failures_for_component
get_documents
trace_entity
get_entity_details
retrieve_historical_evidence

Never request:

get_change_impacts
analyze_change

Never request any IMPACT-* entity.

============================================================
EFFICIENCY
============================================================

Propose at most 8 calls.

Prefer approximately 3 to 6 calls if that is sufficient.

A proposed call should be expected to materially improve:

- recall;
- engineering confidence;
- lifecycle coverage; or
- adjudication quality.

Do not repeat tool calls already present in the evidence
ledger.

============================================================
OUTPUT
============================================================

Return a short explanation followed by:

GAP_PLAN_JSON

{
  "ecr_id": "ECR-001",
  "gaps": [
    {
      "gap_category": "Software / control",
      "reason": "Why this lifecycle gap matters",
      "tool_name": "get_related_components",
      "arguments": {
        "component_id": "BMS-001"
      }
    }
  ]
}

Rules:

- Maximum 8 entries.
- tool_name must be one of the allowed tools.
- arguments must match the selected tool.
- Entity IDs must already be supported by evidence or arise
  from an explicit relationship already present in the
  supplied evidence.
- Do not invent IDs.
- Do not place any text after the JSON.
"""


# ==========================================================
# ALLOWED GAP TOOLS
# ==========================================================

ALLOWED_GAP_TOOLS = {
    "get_component",
    "get_related_components",
    "get_design_specs",
    "get_requirement",
    "search_requirements",
    "get_bom_for_component",
    "get_supplier_for_component",
    "get_manufacturing_for_component",
    "get_validation_test",
    "get_validation_for_requirement",
    "get_service_for_component",
    "get_field_failures_for_component",
    "get_documents",
    "trace_entity",
    "get_entity_details",
    "retrieve_historical_evidence"
}


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


    # Remove an optional Markdown fence.
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
# GAP PLAN VALIDATION
# ==========================================================

def validate_gap_plan(
    gap_plan,
    expected_ecr_id
):

    errors = []


    if gap_plan is None:

        errors.append(
            "Gap plan JSON was not found or could not be parsed."
        )

        return errors


    if not isinstance(
        gap_plan,
        dict
    ):

        errors.append(
            "Gap plan must be a JSON object."
        )

        return errors


    ecr_id = str(
        gap_plan.get(
            "ecr_id",
            ""
        )
    ).strip().upper()


    if ecr_id != str(
        expected_ecr_id
    ).strip().upper():

        errors.append(
            (
                "Gap plan ECR does not match "
                f"{expected_ecr_id}."
            )
        )


    gaps = gap_plan.get(
        "gaps"
    )


    if not isinstance(
        gaps,
        list
    ):

        errors.append(
            "gaps must be a list."
        )

        return errors


    if len(
        gaps
    ) > MAX_GAP_TOOL_CALLS:

        errors.append(
            (
                f"Gap planner requested {len(gaps)} calls. "
                f"Maximum allowed is {MAX_GAP_TOOL_CALLS}."
            )
        )


    for index, gap in enumerate(
        gaps,
        start=1
    ):

        if not isinstance(
            gap,
            dict
        ):

            errors.append(
                f"Gap {index} is not an object."
            )

            continue


        tool_name = gap.get(
            "tool_name"
        )


        arguments = gap.get(
            "arguments"
        )


        reason = gap.get(
            "reason"
        )


        if tool_name not in ALLOWED_GAP_TOOLS:

            errors.append(
                (
                    f"Gap {index} requested prohibited "
                    f"or unknown tool: {tool_name}"
                )
            )


        if not isinstance(
            arguments,
            dict
        ):

            errors.append(
                (
                    f"Gap {index} arguments must be "
                    "a JSON object."
                )
            )


        if not isinstance(
            reason,
            str
        ) or not reason.strip():

            errors.append(
                (
                    f"Gap {index} must include "
                    "a reason."
                )
            )


        # Prevent explicit IMPACT-* access.
        if isinstance(
            arguments,
            dict
        ):

            for value in arguments.values():

                if isinstance(
                    value,
                    str
                ) and is_impact_entity(
                    value
                ):

                    errors.append(
                        (
                            f"Gap {index} attempts to "
                            f"access prohibited entity "
                            f"{value}."
                        )
                    )


    return errors


# ==========================================================
# EXISTING TOOL-CALL KEYS
# ==========================================================

def existing_call_keys(
    evidence_ledger
):

    keys = set()


    for record in evidence_ledger:

        tool_name = record.get(
            "tool"
        )

        arguments = record.get(
            "arguments",
            {}
        )


        keys.add(
            build_tool_cache_key(
                tool_name,
                arguments
            )
        )


    return keys


# ==========================================================
# GAP PLANNER
# ==========================================================

def run_gap_planner(
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
        "Inspect the existing engineering evidence and "
        "identify only high-value lifecycle evidence gaps.\n\n"
        "Do not use ground truth.\n"
        "Do not make the final prediction.\n"
        "Do not repeat existing calls.\n\n"
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
        "D2.1 GAP-DIRECTED RETRIEVAL PLANNER"
    )

    print(
        "=" * 90
    )


    response = client.responses.create(
        model=MODEL,
        instructions=GAP_PLANNER_INSTRUCTIONS,
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


    gap_plan = extract_marker_json(
        response.output_text,
        "GAP_PLAN_JSON"
    )


    errors = validate_gap_plan(
        gap_plan,
        expected_ecr_id=ecr_id
    )


    return {

        "answer":
            response.output_text,

        "gap_plan":
            gap_plan,

        "errors":
            errors,

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
# EXECUTE GAP-DIRECTED CALLS
# ==========================================================

def execute_gap_plan(
    ecr_id,
    gap_plan,
    evidence_ledger,
    benchmark_mode
):

    start_time = time.perf_counter()


    existing_keys = existing_call_keys(
        evidence_ledger
    )


    added_evidence = []

    execution_log = []

    executed_count = 0

    skipped_duplicates = 0

    errors = 0


    gaps = gap_plan.get(
        "gaps",
        []
    )


    for index, gap in enumerate(
        gaps,
        start=1
    ):

        if executed_count >= MAX_GAP_TOOL_CALLS:
            break


        tool_name = gap.get(
            "tool_name"
        )

        arguments = gap.get(
            "arguments",
            {}
        )

        gap_category = gap.get(
            "gap_category",
            "Unspecified"
        )

        reason = gap.get(
            "reason",
            ""
        )


        cache_key = build_tool_cache_key(
            tool_name,
            arguments
        )


        print(
            f"\nGap call {index}: "
            f"{tool_name}"
        )

        print(
            f"Category: {gap_category}"
        )

        print(
            f"Reason: {reason}"
        )

        print(
            f"Arguments: {arguments}"
        )


        # --------------------------------------------------
        # DO NOT REPEAT EXISTING CALLS
        # --------------------------------------------------

        if cache_key in existing_keys:

            skipped_duplicates += 1

            execution_log.append(
                {
                    "tool":
                        tool_name,

                    "arguments":
                        arguments,

                    "gap_category":
                        gap_category,

                    "reason":
                        reason,

                    "status":
                        "duplicate_skipped"
                }
            )

            print(
                "Skipped - identical call already "
                "exists in evidence ledger."
            )

            continue


        # --------------------------------------------------
        # EXECUTE
        # --------------------------------------------------

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


            if benchmark_mode:

                result = (
                    sanitize_benchmark_result(
                        result,
                        active_ecr=ecr_id
                    )
                )


            if (
                isinstance(
                    result,
                    dict
                )
                and result.get(
                    "blocked"
                )
            ):

                status = "security_blocked"


            elif (
                isinstance(
                    result,
                    dict
                )
                and result.get(
                    "_benchmark_withheld"
                )
            ):

                status = "benchmark_withheld"


            elif (
                result_contains_benchmark_withholding(
                    result
                )
            ):

                status = "success_filtered"


            else:

                status = "success"


            executed_count += 1

            existing_keys.add(
                cache_key
            )


            evidence_record = {

                "cycle":
                    "gap-directed",

                "tool":
                    tool_name,

                "arguments":
                    arguments,

                "status":
                    status,

                "gap_category":
                    gap_category,

                "gap_reason":
                    reason,

                "result":
                    result
            }


            evidence_ledger.append(
                evidence_record
            )

            added_evidence.append(
                evidence_record
            )


            execution_log.append(
                {
                    "tool":
                        tool_name,

                    "arguments":
                        arguments,

                    "gap_category":
                        gap_category,

                    "reason":
                        reason,

                    "status":
                        status
                }
            )


            print(
                f"Completed. Status: {status}"
            )


        except Exception as error:

            errors += 1

            execution_log.append(
                {
                    "tool":
                        tool_name,

                    "arguments":
                        arguments,

                    "gap_category":
                        gap_category,

                    "reason":
                        reason,

                    "status":
                        "error",

                    "error":
                        str(
                            error
                        )
                }
            )


            print(
                f"ERROR: {error}"
            )


    elapsed = round(
        time.perf_counter()
        - start_time,
        3
    )


    return {

        "expanded_evidence_ledger":
            evidence_ledger,

        "added_evidence":
            added_evidence,

        "execution_log":
            execution_log,

        "metrics": {

            "gap_calls_executed":
                executed_count,

            "duplicate_calls_skipped":
                skipped_duplicates,

            "errors":
                errors,

            "elapsed_seconds":
                elapsed
        }
    }


# ==========================================================
# SAVE D2.1 RESULT
# ==========================================================

def save_d21_prediction(
    ecr_id,
    benchmark_mode,
    provisional_prediction,
    final_prediction,
    retrieval_metrics,
    planner_metrics,
    gap_execution_metrics,
    adjudication_metrics,
    initial_tool_log,
    gap_execution_log,
    gap_plan,
    adjudication_answer
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
        f"d21_"
        f"{mode_name}_"
        f"{timestamp}.json"
    )


    output_path = (
        D21_PREDICTIONS_FOLDER
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
        + adjudication_metrics.get(
            "total_tokens",
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
        + adjudication_metrics.get(
            "elapsed_seconds",
            0
        ),
        3
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


    payload = {

        "system":
            "Aureon Autonomous AI-PLM Agent",

        "architecture":
            (
                "D2.1 - Gap-Directed Retrieval "
                "+ Impact Adjudication"
            ),

        "phase":
            "5D.1",

        "model":
            MODEL,

        "ecr_id":
            ecr_id,

        "benchmark_mode":
            benchmark_mode,

        "created_at":
            datetime.now().isoformat(),

        "provisional_prediction":
            provisional_prediction,

        "gap_plan":
            gap_plan,

        "prediction":
            final_prediction,

        "retrieval_metrics":
            retrieval_metrics,

        "planner_metrics":
            planner_metrics,

        "gap_execution_metrics":
            gap_execution_metrics,

        "adjudication_metrics":
            adjudication_metrics,

        "combined_metrics": {

            "tool_calls_executed":
                total_tools,

            "total_tokens":
                total_tokens,

            "elapsed_seconds":
                total_elapsed
        },

        "initial_tool_log":
            initial_tool_log,

        "gap_execution_log":
            gap_execution_log,

        "adjudication_answer":
            adjudication_answer
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
# PRINT GAP PLAN
# ==========================================================

def print_gap_plan(
    gap_plan
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.1 GAP PLAN"
    )

    print(
        "=" * 90
    )


    gaps = gap_plan.get(
        "gaps",
        []
    )


    print(
        f"\nPlanned targeted calls: "
        f"{len(gaps)}"
    )


    if not gaps:

        print(
            "  None"
        )

        return


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


# ==========================================================
# PRINT GAP EXECUTION
# ==========================================================

def print_gap_execution(
    execution_log
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.1 GAP-DIRECTED TOOL EXECUTION"
    )

    print(
        "=" * 90
    )


    if not execution_log:

        print(
            "\nNo gap-directed tool calls."
        )

        return


    for index, record in enumerate(
        execution_log,
        start=1
    ):

        print(
            f"\n{index}. "
            f"{record.get('tool')}"
        )

        print(
            f"   Category: "
            f"{record.get('gap_category')}"
        )

        print(
            f"   Status: "
            f"{record.get('status')}"
        )

        print(
            f"   Arguments: "
            f"{record.get('arguments')}"
        )


# ==========================================================
# COMPLETE D2.1 ANALYSIS
# ==========================================================

def analyse_change_d21(
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
        "D2.1 STAGE 1 - FOCUSED AUTONOMOUS RETRIEVAL"
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
    # GAP ANALYSIS
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
                ],

            "gap_planner_answer":
                gap_planner_result[
                    "answer"
                ]
        }


    gap_plan = (
        gap_planner_result[
            "gap_plan"
        ]
    )


    # ======================================================
    # STAGE 3
    # EXECUTE TARGETED GAP CALLS
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
    # FINAL ADJUDICATION
    # ======================================================

    adjudication_result = (
        run_impact_adjudicator(
            ecr_id=ecr_id,
            provisional_answer=
                provisional_answer,
            evidence_ledger=
                expanded_ledger
        )
    )


    final_prediction = (
        extract_prediction_json(
            adjudication_result[
                "answer"
            ]
        )
    )


    final_errors = (
        validate_d2_prediction(
            prediction=
                final_prediction,
            expected_ecr_id=
                ecr_id,
            evidence_ledger=
                expanded_ledger
        )
    )


    if final_errors:

        return {

            "status":
                "final_prediction_failed",

            "errors":
                final_errors,

            "provisional_prediction":
                provisional_prediction,

            "gap_plan":
                gap_plan
        }


    # ======================================================
    # STAGE 5
    # SAVE
    # ======================================================

    output_file = (
        save_d21_prediction(
            ecr_id=ecr_id,
            benchmark_mode=
                benchmark_mode,
            provisional_prediction=
                provisional_prediction,
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
            adjudication_metrics=
                adjudication_result[
                    "metrics"
                ],
            initial_tool_log=
                retrieval_result[
                    "tool_log"
                ],
            gap_execution_log=
                gap_execution[
                    "execution_log"
                ],
            gap_plan=
                gap_plan,
            adjudication_answer=
                adjudication_result[
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

        "final_answer":
            adjudication_result[
                "answer"
            ],

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

        "adjudication_metrics":
            adjudication_result[
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
        "Phase 5D.1 - D2.1 Gap-Directed Retrieval"
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
        "  Stage 2 - Lifecycle gap planner"
    )

    print(
        "  Stage 3 - Targeted evidence acquisition"
    )

    print(
        "  Stage 4 - Independent impact adjudication"
    )


    print(
        f"\nInitial retrieval tool budget: "
        f"{INITIAL_TOOL_BUDGET}"
    )

    print(
        f"Maximum gap-directed calls: "
        f"{MAX_GAP_TOOL_CALLS}"
    )

    print(
        f"Maximum planned total calls: "
        f"{MAX_TOTAL_TOOL_CALLS}"
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
        "\nFor Phase 5D.1, use ECR-001 "
        "as the development case."
    )


    result = analyse_change_d21(
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
            "D2.1 ANALYSIS FAILED"
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


        sys.exit(
            1
        )


    # ======================================================
    # PROVISIONAL SET
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
        "D2.1 PROVISIONAL PREDICTION"
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

    print_gap_plan(
        result[
            "gap_plan"
        ]
    )


    # ======================================================
    # GAP EXECUTION
    # ======================================================

    print_gap_execution(
        result[
            "gap_execution_log"
        ]
    )


    # ======================================================
    # FINAL ADJUDICATION
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.1 FINAL ADJUDICATED RESPONSE"
    )

    print(
        "=" * 90
    )


    print(
        result[
            "final_answer"
        ]
    )


    # ======================================================
    # FINAL SET
    # ======================================================

    final_entities = [
        item[
            "entity_id"
        ]
        for item in result[
            "final_prediction"
        ][
            "predicted_impacts"
        ]
    ]


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.1 FINAL PREDICTION"
    )

    print(
        "=" * 90
    )


    print(
        "\nPrediction validation PASSED."
    )


    print(
        f"\nFinal predicted entities: "
        f"{len(final_entities)}"
    )


    for entity_id in final_entities:

        print(
            f"  - {entity_id}"
        )


    # ======================================================
    # DELTA
    # ======================================================

    provisional_set = set(
        provisional_entities
    )

    final_set = set(
        final_entities
    )


    removed = sorted(
        provisional_set
        - final_set
    )


    added = sorted(
        final_set
        - provisional_set
    )


    print(
        "\nAdjudication delta:"
    )


    print(
        f"\nRemoved: {len(removed)}"
    )


    if removed:

        for entity in removed:

            print(
                f"  - {entity}"
            )

    else:

        print(
            "  None"
        )


    print(
        f"\nAdded after gap-directed retrieval: "
        f"{len(added)}"
    )


    if added:

        for entity in added:

            print(
                f"  - {entity}"
            )

    else:

        print(
            "  None"
        )


    # ======================================================
    # METRICS
    # ======================================================

    retrieval_metrics = (
        result[
            "retrieval_metrics"
        ]
    )

    planner_metrics = (
        result[
            "planner_metrics"
        ]
    )

    gap_metrics = (
        result[
            "gap_execution_metrics"
        ]
    )

    adjudication_metrics = (
        result[
            "adjudication_metrics"
        ]
    )


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
        + adjudication_metrics.get(
            "total_tokens",
            0
        )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2.1 EXECUTION METRICS"
    )

    print(
        "=" * 90
    )


    print(
        "\nInitial retrieval:"
    )

    print(
        f"  Tools executed: "
        f"{retrieval_metrics.get('tool_calls_executed')}"
    )

    print(
        f"  Reasoning cycles: "
        f"{retrieval_metrics.get('reasoning_cycles')}"
    )

    print(
        f"  Tokens: "
        f"{retrieval_metrics.get('total_tokens')}"
    )


    print(
        "\nGap planner:"
    )

    print(
        f"  Tokens: "
        f"{planner_metrics.get('total_tokens')}"
    )


    print(
        "\nTargeted retrieval:"
    )

    print(
        f"  Calls executed: "
        f"{gap_metrics.get('gap_calls_executed')}"
    )

    print(
        f"  Duplicate calls skipped: "
        f"{gap_metrics.get('duplicate_calls_skipped')}"
    )

    print(
        f"  Errors: "
        f"{gap_metrics.get('errors')}"
    )


    print(
        "\nFinal adjudication:"
    )

    print(
        f"  Tokens: "
        f"{adjudication_metrics.get('total_tokens')}"
    )


    print(
        "\nCombined:"
    )

    print(
        f"  Total PLM tool calls: "
        f"{total_tools}"
    )

    print(
        f"  Total model tokens: "
        f"{total_tokens}"
    )

    print(
        f"  Overall elapsed time: "
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
        "PHASE 5D.1 COMPLETE"
    )

    print(
        "=" * 90
    )