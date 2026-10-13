# ==========================================================
# AUREON AUTONOMOUS AI-PLM AGENT
# Phase 5D.0
#
# D2 Architecture:
# Autonomous Retrieval + Impact Adjudication Layer
#
# IMPORTANT:
# D1 remains frozen in:
# agent/tool_calling_agent.py
# ==========================================================


# ==========================================================
# WINDOWS DLL INITIALIZATION
# ==========================================================

# Keep PyTorch first because of the Windows DLL / c10.dll
# behaviour already observed in this project.
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
# REUSE THE FROZEN D1 INFRASTRUCTURE
# ==========================================================

from agent.tool_calling_agent import (
    client,
    MODEL,
    TOOLS,
    AGENT_INSTRUCTIONS,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    execute_tool,
    clean_json_value,
    sanitize_benchmark_result,
    update_usage_metrics,
    build_tool_cache_key,
    result_contains_benchmark_withholding,
    extract_prediction_json,
    validate_prediction,
    is_impact_entity
)


# ==========================================================
# D2 OUTPUT FOLDER
# ==========================================================

D2_PREDICTIONS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "predictions_d2"
)

D2_PREDICTIONS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# D2 RETRIEVAL / REASONING INSTRUCTIONS
# ==========================================================

D2_AGENT_INSTRUCTIONS = (
    AGENT_INSTRUCTIONS
    + """

============================================================
D2 ARCHITECTURE — PROVISIONAL IMPACT GENERATION
============================================================

You are operating as the evidence-retrieval and provisional
impact-reasoning stage of the D2 Aureon PLM Agent.

Your PREDICTION_JSON at this stage is PROVISIONAL.

A separate impact-adjudication layer will review your
prediction after tool use is complete.

Therefore:

1. Gather the strongest evidence necessary for engineering
   change-impact reasoning.

2. Do not intentionally over-expand the prediction merely
   to improve recall.

3. Do not omit an engineering domain merely because the
   directly changed component is already understood.

4. Before finishing, perform an internal lifecycle
   completeness check.

============================================================
D2 LIFECYCLE COMPLETENESS CHECK
============================================================

Before producing the provisional prediction, explicitly ask
whether sufficient evidence has been considered for each
relevant domain:

- primary changed component
- directly connected components
- mechanical structure / mounting
- electrical / power path
- thermal system
- software / control
- requirements
- validation / verification
- manufacturing
- supplier
- service / maintainability
- configuration / documentation

Not every engineering change affects every domain.

The objective is NOT to force an impact into every category.

The objective is to prevent important lifecycle categories
from being forgotten simply because they are one or two
relationships away from the primary component.

============================================================
D2 PROPAGATION DISCIPLINE
============================================================

When considering second-order impacts:

- follow explicit engineering relationships;
- consider plausible engineering consequences;
- seek supporting PLM evidence where available;
- stop when propagation becomes speculative.

Do not infer an entity merely because a category should be
checked.

============================================================
D2 RECORD-TYPE DISCIPLINE
============================================================

Be especially careful with:

- TEST-*
- MFG-*
- SRV-*
- SUP-*
- BOM-*
- DOC-*
- SPEC-*
- PROD-*

Retrieving one of these records does NOT automatically mean
that it is an impacted entity.

A record may be:

- evidence only;
- context only;
- directly impacted;
- review-required.

The later adjudication layer will make the final decision.

============================================================
D2 PROVISIONAL OUTPUT
============================================================

Keep the normal engineering narrative and PREDICTION_JSON.

The PREDICTION_JSON represents your provisional impact set.

Do not place text after the PREDICTION_JSON.
"""
)


# ==========================================================
# D2 ADJUDICATOR INSTRUCTIONS
# ==========================================================

ADJUDICATOR_INSTRUCTIONS = """
You are the independent Impact Adjudication Layer of the
Aureon Autonomous AI-PLM Agent D2.

The autonomous retrieval agent has already:

- analysed one engineering change;
- called permitted PLM tools;
- gathered engineering evidence;
- produced a provisional impact prediction.

You may NOT call tools.

You may NOT access ground truth.

You must adjudicate using only:

1. the active ECR;
2. the retrieved evidence ledger;
3. the provisional engineering report.

============================================================
PRIMARY OBJECTIVE
============================================================

Separate:

RELATED

from:

GENUINELY IMPACTED OR ENGINEERING-REVIEW-REQUIRED.

The fact that an entity was retrieved does not mean that it
belongs in the final impact set.

============================================================
ADJUDICATION CLASSES
============================================================

Internally classify candidate entities as:

1. IMPACTED
   Evidence strongly supports that the change alters,
   invalidates, revises, triggers, or directly affects the
   entity.

2. REVIEW-REQUIRED
   The entity is not confirmed to change, but engineering
   compatibility, verification, qualification, or lifecycle
   disposition is materially necessary because of the
   engineering change.

3. CONTEXT-ONLY
   The entity helped understand the change but does not
   itself need to be treated as impacted.

4. NOT-SUPPORTED
   Available evidence is insufficient to justify including
   the entity.

Only IMPACTED and REVIEW-REQUIRED entities may appear in the
final PREDICTION_JSON.

============================================================
EVIDENCE REQUIREMENT
============================================================

An entity may appear in the final prediction only if its
entity ID is explicitly present in:

- the active ECR evidence;
- a structured tool result;
- an explicit relationship result;
- a validation/manufacturing/service/supplier result;
- or permitted historical evidence.

Never invent an entity ID.

Never add an entity merely because an engineering domain
should theoretically contain one.

============================================================
LIFECYCLE COMPLETENESS AUDIT
============================================================

Before finalizing, check whether the gathered evidence
supports relevant impacts in:

- primary component
- component dependencies
- mechanical / structural
- electrical / power electronics
- thermal
- software / control
- requirements
- validation
- manufacturing
- supplier
- service
- configuration / documentation

This is a completeness audit, not a requirement to predict
something from every category.

============================================================
PROPAGATION RULE
============================================================

Use a controlled propagation hierarchy:

Direct
→ First-order
→ Second-order
→ Lifecycle

Do not propagate indefinitely.

A second-order entity must have an intelligible engineering
path from the active change.

============================================================
REQUIREMENT RULE
============================================================

Include a requirement when:

- its stored baseline directly conflicts with the change;
- the changed design materially affects its satisfaction;
- or the change requires its engineering verification.

Do not include a requirement merely because it concerns the
same broad subsystem.

============================================================
VALIDATION RULE
============================================================

Include a TEST-* entity only when the available evidence
shows that the test verifies a requirement or design aspect
materially affected by the changed configuration.

Do not include every nearby test.

Baseline test success does not validate a changed
configuration.

============================================================
MANUFACTURING RULE
============================================================

Include an MFG-* entity when the engineering change is
reasonably expected to alter, requalify, review, or directly
affect that manufacturing operation.

Do not include a manufacturing record merely because it
produces the same component.

============================================================
SERVICE RULE
============================================================

Include an SRV-* entity when the engineering change has a
supported serviceability, maintenance, field-action,
diagnostic, replacement, or service-record consequence.

Do not create a service impact merely because the component
has service records.

============================================================
SUPPLIER RULE
============================================================

Include a SUP-* entity when sourcing, qualification,
approval, specification, or supplier deliverables are
materially implicated.

An existing approved supplier is not automatically an
impact.

============================================================
BOM / DOCUMENT / PRODUCT RULE
============================================================

Include BOM-*, DOC-* or PROD-* only when that specific PLM
object itself requires lifecycle disposition because of the
change.

Do not promote these entities merely because:

- the component appears in a BOM;
- a document exists;
- or the component belongs to the product.

============================================================
SPECIFICATION RULE
============================================================

Include SPEC-* only if the specification record itself is
materially changed, invalidated, or requires revision /
verification.

Do not include SPEC-* simply because it was consulted.

============================================================
CONFIDENCE
============================================================

Use:

Confirmed
Strong candidate
Review candidate

Confirmed requires explicit evidence.

Strong candidate requires a strong engineering path and
supporting evidence.

Review candidate requires a meaningful compatibility or
lifecycle check supported by the evidence.

============================================================
FINAL RESPONSE
============================================================

Provide a concise section:

D2 IMPACT ADJUDICATION SUMMARY

State:

- what was retained;
- what was removed as context-only or insufficient;
- whether any evidence-supported entity was added during the
  lifecycle completeness audit;
- major remaining uncertainty.

Then output:

PREDICTION_JSON

followed by valid JSON:

{
  "ecr_id": "ECR-001",
  "predicted_impacts": [
    {
      "entity_id": "BAT-001",
      "impact_level": "Direct",
      "confidence": "Confirmed",
      "reason": "Evidence-grounded reason."
    }
  ]
}

Allowed impact_level values:

Direct
First-order
Second-order
Lifecycle

Allowed confidence values:

Confirmed
Strong candidate
Review candidate

Every entity must appear only once.

Never include IMPACT-* entities.

Do not place any text after PREDICTION_JSON.
"""


# ==========================================================
# METRIC HELPERS
# ==========================================================

def new_metrics():
    return {
        "reasoning_cycles": 0,
        "tool_calls_requested": 0,
        "tool_calls_executed": 0,
        "blocked_calls": 0,
        "benchmark_withheld_results": 0,
        "cache_hits": 0,
        "tool_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "elapsed_seconds": 0.0
    }


def usage_dict(
    response
):
    """
    Extract token usage from one OpenAI response.
    """

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

    result["input_tokens"] = (
        getattr(
            usage,
            "input_tokens",
            0
        )
        or 0
    )

    result["output_tokens"] = (
        getattr(
            usage,
            "output_tokens",
            0
        )
        or 0
    )

    result["total_tokens"] = (
        getattr(
            usage,
            "total_tokens",
            0
        )
        or 0
    )

    return result


# ==========================================================
# EVIDENCE LEDGER
# ==========================================================

def add_to_evidence_ledger(
    ledger,
    cycle,
    tool_name,
    arguments,
    result,
    status
):
    """
    Store the actual evidence seen by the retrieval agent.

    The D2 adjudicator sees this ledger but has no tool
    access of its own.
    """

    ledger.append(
        {
            "cycle": cycle,
            "tool": tool_name,
            "arguments": arguments,
            "status": status,
            "result": result
        }
    )


# ==========================================================
# D2 RETRIEVAL AGENT
# ==========================================================

def run_retrieval_agent(
    ecr_id,
    benchmark_mode=False,
    max_iterations=MAX_ITERATIONS,
    max_tool_calls=MAX_TOOL_CALLS
):
    """
    Run the autonomous evidence-retrieval stage.

    This resembles D1 but additionally stores the complete
    permitted evidence ledger for the D2 adjudicator.
    """

    start_time = time.perf_counter()

    ecr_id = (
        str(ecr_id)
        .strip()
        .upper()
    )

    metrics = new_metrics()

    tool_log = []

    evidence_ledger = []

    tool_cache = {}


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

This is the D2 retrieval stage.

Start by retrieving the engineering change record.

Then inspect the active ECR lifecycle trace unless filtered
by benchmark mode.

Gather enough evidence for both:

1. accurate impact detection; and
2. later independent impact adjudication.

Before finishing, perform the lifecycle completeness check
defined in your instructions.

Do not call every tool.

Do not retrieve records merely to fill categories.

Never request IMPACT-* entities.

Never access evaluator ground truth.

Do not use get_change_impacts or analyze_change.

In benchmark mode, accept withheld evidence and do not try
to reconstruct it.

Produce the normal engineering report and a provisional
PREDICTION_JSON when evidence gathering is complete.
"""


    input_items = [
        {
            "role": "user",
            "content": initial_prompt
        }
    ]


    for iteration in range(
        1,
        max_iterations + 1
    ):

        metrics[
            "reasoning_cycles"
        ] = iteration


        print(
            f"\n--- D2 retrieval reasoning cycle "
            f"{iteration} ---"
        )


        response = client.responses.create(
            model=MODEL,
            instructions=D2_AGENT_INSTRUCTIONS,
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
        # RETRIEVAL STAGE FINISHED
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
                "answer": response.output_text,
                "tool_log": tool_log,
                "evidence_ledger": evidence_ledger,
                "metrics": metrics,
                "iterations": iteration
            }


        # ==================================================
        # PROCESS FUNCTION CALLS
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
                f"Tool selected: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )


            cache_key = (
                build_tool_cache_key(
                    tool_name,
                    arguments
                )
            )


            # ==============================================
            # CACHE HIT
            # ==============================================

            if cache_key in tool_cache:

                result = tool_cache[
                    cache_key
                ]

                metrics[
                    "cache_hits"
                ] += 1

                status = "cache_hit"


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
                    "blocked": True,
                    "reason": (
                        "Tool-call budget exhausted. "
                        "Use the evidence already collected."
                    )
                }

                metrics[
                    "blocked_calls"
                ] += 1

                status = "budget_blocked"


            # ==============================================
            # EXECUTION
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

                        metrics[
                            "blocked_calls"
                        ] += 1

                        status = (
                            "security_blocked"
                        )


                    elif (
                        isinstance(
                            result,
                            dict
                        )
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

                        status = "success"


                    tool_cache[
                        cache_key
                    ] = result


                except Exception as error:

                    result = {
                        "error": str(
                            error
                        )
                    }

                    metrics[
                        "tool_errors"
                    ] += 1

                    status = "error"


            tool_log.append(
                {
                    "cycle": iteration,
                    "tool": tool_name,
                    "arguments": arguments,
                    "status": status
                }
            )


            # Store actual evidence returned.
            add_to_evidence_ledger(
                ledger=evidence_ledger,
                cycle=iteration,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                status=status
            )


            print(
                f"Tool completed. Status: {status}"
            )


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
    # SAFETY STOP
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
            "D2 retrieval stage stopped after reaching "
            f"{max_iterations} reasoning cycles."
        ),
        "tool_log": tool_log,
        "evidence_ledger": evidence_ledger,
        "metrics": metrics,
        "iterations": max_iterations
    }


# ==========================================================
# D2 ADJUDICATION
# ==========================================================

def run_impact_adjudicator(
    ecr_id,
    provisional_answer,
    evidence_ledger
):
    """
    Second model pass.

    No tools.
    No ground truth.
    Only the evidence already retrieved by the agent.
    """

    start_time = time.perf_counter()


    adjudication_payload = {
        "active_ecr": ecr_id,
        "provisional_report": provisional_answer,
        "evidence_ledger": evidence_ledger
    }


    user_prompt = (
        "Adjudicate the following D2 provisional "
        "engineering-change assessment.\n\n"
        "Do not call tools.\n"
        "Do not use outside ground truth.\n"
        "Use only the supplied evidence.\n\n"
        + json.dumps(
            adjudication_payload,
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
        "D2 IMPACT ADJUDICATION LAYER"
    )

    print(
        "=" * 90
    )


    response = client.responses.create(
        model=MODEL,
        instructions=ADJUDICATOR_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": user_prompt
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


    metrics = {
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


    return {
        "answer":
            response.output_text,

        "metrics":
            metrics
    }


# ==========================================================
# FINAL D2 PREDICTION VALIDATION
# ==========================================================

def validate_d2_prediction(
    prediction,
    expected_ecr_id,
    evidence_ledger
):
    """
    Run the normal D1 prediction validation plus a D2
    evidence-presence check.

    Every predicted entity must have appeared somewhere in
    the evidence actually available to the adjudicator.
    """

    errors = validate_prediction(
        prediction,
        expected_ecr_id
    )


    if prediction is None:
        return errors


    evidence_text = json.dumps(
        evidence_ledger,
        ensure_ascii=False,
        default=str
    ).upper()


    impacts = prediction.get(
        "predicted_impacts",
        []
    )


    for impact in impacts:

        if not isinstance(
            impact,
            dict
        ):
            continue


        entity_id = str(
            impact.get(
                "entity_id",
                ""
            )
        ).strip().upper()


        if not entity_id:
            continue


        if is_impact_entity(
            entity_id
        ):

            errors.append(
                (
                    f"{entity_id} is prohibited "
                    "in D2 predictions."
                )
            )

            continue


        if entity_id not in evidence_text:

            errors.append(
                (
                    f"{entity_id} was predicted by the "
                    "adjudicator but does not appear in "
                    "the retrieved evidence ledger."
                )
            )


    return errors


# ==========================================================
# SAVE D2 RUN
# ==========================================================

def save_d2_prediction(
    ecr_id,
    benchmark_mode,
    provisional_prediction,
    final_prediction,
    retrieval_metrics,
    adjudication_metrics,
    tool_log,
    adjudication_answer
):
    """
    Save D2 results separately from D1.
    """

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
        f"d2_"
        f"{mode_name}_"
        f"{timestamp}.json"
    )


    output_path = (
        D2_PREDICTIONS_FOLDER
        / filename
    )


    combined_total_tokens = (
        retrieval_metrics.get(
            "total_tokens",
            0
        )
        + adjudication_metrics.get(
            "total_tokens",
            0
        )
    )


    combined_elapsed = round(
        retrieval_metrics.get(
            "elapsed_seconds",
            0
        )
        + adjudication_metrics.get(
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
                "D2 - Autonomous Retrieval + "
                "Impact Adjudication Layer"
            ),

        "phase":
            "5D.0",

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

        "prediction":
            final_prediction,

        "retrieval_metrics":
            retrieval_metrics,

        "adjudication_metrics":
            adjudication_metrics,

        "combined_metrics": {
            "total_tokens":
                combined_total_tokens,

            "elapsed_seconds":
                combined_elapsed,

            "tool_calls_executed":
                retrieval_metrics.get(
                    "tool_calls_executed",
                    0
                ),

            "reasoning_cycles":
                retrieval_metrics.get(
                    "reasoning_cycles",
                    0
                )
        },

        "tool_log":
            tool_log,

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
# DISPLAY TOOL TRACE
# ==========================================================

def print_tool_trace(
    tool_log
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2 RETRIEVAL TOOL TRACE"
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
# DISPLAY METRICS
# ==========================================================

def print_metrics(
    retrieval_metrics,
    adjudication_metrics
):

    combined_tokens = (
        retrieval_metrics.get(
            "total_tokens",
            0
        )
        + adjudication_metrics.get(
            "total_tokens",
            0
        )
    )


    combined_time = (
        retrieval_metrics.get(
            "elapsed_seconds",
            0
        )
        + adjudication_metrics.get(
            "elapsed_seconds",
            0
        )
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2 EXECUTION METRICS"
    )

    print(
        "=" * 90
    )


    print(
        "\nRetrieval stage:"
    )

    print(
        "  Reasoning cycles: "
        f"{retrieval_metrics.get('reasoning_cycles')}"
    )

    print(
        "  Tool requests: "
        f"{retrieval_metrics.get('tool_calls_requested')}"
    )

    print(
        "  Tools executed: "
        f"{retrieval_metrics.get('tool_calls_executed')}"
    )

    print(
        "  Retrieval tokens: "
        f"{retrieval_metrics.get('total_tokens')}"
    )

    print(
        "  Retrieval time: "
        f"{retrieval_metrics.get('elapsed_seconds')} s"
    )


    print(
        "\nAdjudication stage:"
    )

    print(
        "  Input tokens: "
        f"{adjudication_metrics.get('input_tokens')}"
    )

    print(
        "  Output tokens: "
        f"{adjudication_metrics.get('output_tokens')}"
    )

    print(
        "  Total tokens: "
        f"{adjudication_metrics.get('total_tokens')}"
    )

    print(
        "  Adjudication time: "
        f"{adjudication_metrics.get('elapsed_seconds')} s"
    )


    print(
        "\nCombined D2:"
    )

    print(
        f"  Total tokens: {combined_tokens}"
    )

    print(
        f"  Total elapsed time: "
        f"{combined_time:.3f} s"
    )


# ==========================================================
# COMPLETE D2 ANALYSIS
# ==========================================================

def analyse_change_d2(
    ecr_id,
    benchmark_mode=False
):
    """
    Complete D2 pipeline:

    Retrieval
       ↓
    Provisional prediction
       ↓
    Adjudication
       ↓
    Final validated prediction
    """

    ecr_id = (
        str(ecr_id)
        .strip()
        .upper()
    )


    # ======================================================
    # STAGE 1 — AUTONOMOUS RETRIEVAL
    # ======================================================

    retrieval_result = (
        run_retrieval_agent(
            ecr_id=ecr_id,
            benchmark_mode=benchmark_mode
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
                provisional_errors,

            "retrieval_result":
                retrieval_result
        }


    # ======================================================
    # STAGE 2 — IMPACT ADJUDICATION
    # ======================================================

    adjudication_result = (
        run_impact_adjudicator(
            ecr_id=ecr_id,
            provisional_answer=
                provisional_answer,
            evidence_ledger=
                retrieval_result[
                    "evidence_ledger"
                ]
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
            prediction=final_prediction,
            expected_ecr_id=ecr_id,
            evidence_ledger=
                retrieval_result[
                    "evidence_ledger"
                ]
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

            "retrieval_result":
                retrieval_result,

            "adjudication_result":
                adjudication_result
        }


    # ======================================================
    # STAGE 3 — SAVE
    # ======================================================

    output_file = (
        save_d2_prediction(
            ecr_id=ecr_id,
            benchmark_mode=benchmark_mode,
            provisional_prediction=
                provisional_prediction,
            final_prediction=
                final_prediction,
            retrieval_metrics=
                retrieval_result[
                    "metrics"
                ],
            adjudication_metrics=
                adjudication_result[
                    "metrics"
                ],
            tool_log=
                retrieval_result[
                    "tool_log"
                ],
            adjudication_answer=
                adjudication_result[
                    "answer"
                ]
        )
    )


    return {
        "status":
            "success",

        "ecr_id":
            ecr_id,

        "benchmark_mode":
            benchmark_mode,

        "provisional_answer":
            provisional_answer,

        "provisional_prediction":
            provisional_prediction,

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

        "adjudication_metrics":
            adjudication_result[
                "metrics"
            ],

        "tool_log":
            retrieval_result[
                "tool_log"
            ],

        "prediction_file":
            output_file
    }


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
        "Phase 5D.0 - D2 Impact Adjudication Architecture"
    )

    print(
        "=" * 90
    )


    print(
        f"\nModel: {MODEL}"
    )


    print(
        "\nD1 baseline remains unchanged:"
    )

    print(
        "  agent/tool_calling_agent.py"
    )


    print(
        "\nD2 adds:"
    )

    print(
        "  Autonomous evidence retrieval"
    )

    print(
        "  Lifecycle completeness audit"
    )

    print(
        "  Independent impact adjudication"
    )

    print(
        "  Evidence-presence validation"
    )


    ecr_id = input(
        "\nEnter ECR ID "
        "(development test: ECR-001): "
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
        "\nSelect mode [2]: "
    ).strip()


    if not mode:
        mode = "2"


    benchmark_mode = (
        mode == "2"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "For Phase 5D.0 use ECR-001 as the development "
        "case."
    )

    print(
        "Do not use the previously inspected nine held-out "
        "cases as a new headline test set."
    )


    result = analyse_change_d2(
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
            "D2 ANALYSIS FAILED"
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
    # TOOL TRACE
    # ======================================================

    print_tool_trace(
        result[
            "tool_log"
        ]
    )


    # ======================================================
    # PROVISIONAL PREDICTION
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2 PROVISIONAL PREDICTION"
    )

    print(
        "=" * 90
    )


    provisional_entities = [
        impact[
            "entity_id"
        ]
        for impact in result[
            "provisional_prediction"
        ][
            "predicted_impacts"
        ]
    ]


    print(
        f"\nProvisional entities: "
        f"{len(provisional_entities)}"
    )


    for entity_id in provisional_entities:

        print(
            f"  - {entity_id}"
        )


    # ======================================================
    # FINAL ADJUDICATED RESPONSE
    # ======================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "D2 FINAL ADJUDICATED RESPONSE"
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
    # FINAL ENTITY SET
    # ======================================================

    final_entities = [
        impact[
            "entity_id"
        ]
        for impact in result[
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
        "D2 FINAL PREDICTION STATUS"
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
    # ADJUDICATION DELTA
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
        "\nAdjudication changes:"
    )


    print(
        f"\nRemoved from provisional set: "
        f"{len(removed)}"
    )


    if removed:

        for entity_id in removed:

            print(
                f"  - {entity_id}"
            )

    else:

        print(
            "  None"
        )


    print(
        f"\nAdded during completeness audit: "
        f"{len(added)}"
    )


    if added:

        for entity_id in added:

            print(
                f"  - {entity_id}"
            )

    else:

        print(
            "  None"
        )


    # ======================================================
    # METRICS
    # ======================================================

    print_metrics(
        retrieval_metrics=
            result[
                "retrieval_metrics"
            ],
        adjudication_metrics=
            result[
                "adjudication_metrics"
            ]
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
        "PHASE 5D.0 D2 ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )