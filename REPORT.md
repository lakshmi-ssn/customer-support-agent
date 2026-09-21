# HW2 Report — <your name>, <roll number>

## The short version

|                                    | Score / measurement |
| ---------------------------------- | ------------------: |
| Shipped baseline                   |         70.31 / 100 |
| TODO 4 result (first recorded run) |         71.67 / 100 |
| Route-fix checkpoint               |         77.40 / 100 |
| Final tuned run                   |         80.00 / 100 |
| Dense retrieval recall@4           |       19/29 = 65.5% |
| Hybrid retrieval recall@4          |       20/29 = 69.0% |

I used model `openai/gpt-4o-mini`, with chunks of 600 characters. The API cost was not recorded in this run: `<add the amount from OpenRouter usage>`.

The latest retrieval-only experiment compared dense retrieval with hybrid retrieval using BM25 lexical search and Reciprocal Rank Fusion (RRF). Dense retrieval achieved 19/29 (65.5%) recall@4, while hybrid retrieval achieved 20/29 (69.0%), an improvement of 1 case or 3.5 percentage points.

## What I changed, and what each change was worth

| What I did                                           |    Result / measurement | Kept it?            |
| ---------------------------------------------------- | ----------------------: | ------------------- |
| Starting point / shipped baseline                    |             70.31 / 100 | yes                 |
| TODO 4 — tool-calling loop                           |             71.67 / 100 | yes                 |
| TODO 2 — dense retrieval measurement                 |  19/29 = 65.5% recall@4 | comparison baseline |
| TODO 2 — hybrid retrieval measurement                |  20/29 = 69.0% recall@4 | yes                 |
| TODO 3 — answer verification against handbook claims | not measured separately | —                   |
| TODO 6 — branches, memory, human approval            | not measured separately | —                   |
| TODO 5 — fake-instruction defense                    | not measured separately | —                   |
| TODO 1 — structured handbook splitting               | not measured separately | —                   |

The retrieval experiment shows a modest improvement from hybrid retrieval. The gain is 1 additional successful retrieval case out of 29, equivalent to 3.5 percentage points.

This also shows that retrieval quality is only one part of the complete agent. Several categories remained unchanged between dense and hybrid retrieval, so routing, tool selection, verification, and escalation logic still need to be handled separately.

## TODO 4 observation: tool-calling loop

The recorded result after implementing the tool-calling loop was 71.67 / 100 on the practice evaluation.

The tool-calling loop allows the model to request available tools and receive their results before producing its response. However, the loop itself does not determine whether a case should be resolved, escalated, or rejected as unsupported. Those decisions require the verification and flow-control logic implemented in later TODOs.

## 1. Searching the handbook

The handbook is loaded by `support_agent.kb.load_sections()`, which reads Markdown files under `config.KB_DIR`, currently `data/kb/handbook.md`.

The retrieval configuration is:

* `TOP_K=4`
* `CANDIDATE_K=8`
* `CHUNK_SIZE=600`
* `CHUNK_OVERLAP=150`
* `RRF_K=60`

Two retrieval approaches were measured.

### Dense retrieval

Dense retrieval searches by semantic similarity. The measurement command was:

```text
$env:RETRIEVAL_MODE="dense"; python scripts/evaluate_dev.py --retrieval-only
```

Result:

```text
retrieval recall@4 (mode=dense): 19/29 = 0.655

  refund_within_limit        0/2   0.00
  cancellation               2/4   0.50
  refund_needs_approval      2/4   0.50
  safety_incident            2/4   0.50
  escalate_other             1/2   0.50
  return_eligibility         2/3   0.67
  policy_qa                  2/2   1.00
  stale_policy               2/2   1.00
  injection                  4/4   1.00
  multi_turn                 2/2   1.00
```

Overall recall@4 was **19/29 = 65.5%**.

### Hybrid retrieval

Hybrid retrieval combines dense retrieval with BM25 lexical retrieval and combines their rankings using Reciprocal Rank Fusion (RRF).

The measurement command was:

```text
$env:RETRIEVAL_MODE="hybrid"; python scripts/evaluate_dev.py --retrieval-only
```

Result:

```text
retrieval recall@4 (mode=hybrid): 20/29 = 0.690

  refund_within_limit        0/2   0.00
  stale_policy               1/2   0.50
  refund_needs_approval      2/4   0.50
  safety_incident            2/4   0.50
  return_eligibility         2/3   0.67
  cancellation               3/4   0.75
  policy_qa                  2/2   1.00
  escalate_other             2/2   1.00
  injection                  4/4   1.00
  multi_turn                 2/2   1.00
```

Overall recall@4 was **20/29 = 69.0%**.

### Dense vs hybrid

| Retrieval mode |      Recall@4 | Result                 |
| -------------- | ------------: | ---------------------- |
| Dense          | 19/29 = 65.5% | baseline               |
| Hybrid         | 20/29 = 69.0% | +1 case                |
| Improvement    |             — | +3.5 percentage points |

The hybrid approach therefore produced a measurable but modest improvement.

The largest category improvements were:

* `cancellation`: 2/4 → 3/4
* `escalate_other`: 1/2 → 2/2

Some categories did not improve:

* `refund_within_limit`: 0/2 → 0/2
* `refund_needs_approval`: 2/4 → 2/4
* `safety_incident`: 2/4 → 2/4
* `return_eligibility`: 2/3 → 2/3
* `policy_qa`: 2/2 → 2/2
* `injection`: 4/4 → 4/4
* `multi_turn`: 2/2 → 2/2

Interestingly, `stale_policy` decreased from 2/2 to 1/2 in the hybrid run. Therefore, the experiment does not support a claim that hybrid retrieval improves every category.

The configured untrusted section is `community`, and the superseded section is `archive_returns_2024`. These sections must not be used as trusted answer sources.

The retrieval post-processing removes hits whose document IDs appear in `UNTRUSTED_DOCS` or `SUPERSEDED_DOCS` before they are passed forward as trusted retrieval results.

## 2. Checking the answer is true

The current evaluation score for citations was 0.708, or 70.8%, across 24 queries in the recorded evaluation run.

A separate measurement of the claim-checking cutoff, added latency, and token cost was not recorded in that run.

The intended design is that retrieved handbook evidence is used to ground the response and that unsupported claims should not be presented as handbook-backed facts.

### Claim-level verification

Implemented `graph.node_verify` using the Lecture 6 faithfulness approach:

1. Decompose the generated answer into atomic factual claims.
2. Check each claim against the retrieved handbook context.
3. Calculate the faithfulness score as:

   `sum(supported_claims) / number_of_claims`

4. If any claim is unsupported, the agent escalates rather than returning the unverified answer.

Isolation tests:

- Supported claim: `Returns are allowed within the return window.`
  - Faithfulness score: `1.0`
  - Supported: `True`
  - Result: verification passed.

- Unsupported claim: `Customers get a free laptop with every return.`
  - Faithfulness score: `0.0`
  - Supported: `False`
  - Result: `escalated`.

This demonstrates that the verifier distinguishes supported from unsupported claims. The verifier has not yet been evaluated as part of the complete graph flow.

## 3. Tools and safety

The evaluation reported zero safety violations in the recorded runs. However, the overall score remained limited by incorrect routes and missing required actions, particularly for refund and escalation cases.

The recorded weak categories included:

* `refund_within_limit`
* `escalate_other`
* `injection`
* `refund_needs_approval`
* `safety_incident`

The recorded worst cases included:

* `dev-008`: route escalated instead of resolved; missing `create_return`.
* `dev-011`: route escalated instead of resolved; missing `issue_wallet_credit`.
* `dev-012`: route escalated instead of resolved; missing `issue_wallet_credit`.
* `dev-013`: missing `escalate_to_human`.
* `dev-014`: missing `escalate_to_human`.
* `dev-017`: missing `escalate_to_human`.
* `dev-018`: missing `escalate_to_human`.
* `dev-019`: missing `escalate_to_human`.

These failures indicate that having the tools available is not sufficient: the agent must select the correct tool, supply the required arguments, and follow the handbook's escalation rules.

## 4. Controlling the flow

The original graph followed a straight path through:

```text
lookup → retrieve → respond → END
```

The graph therefore did not initially contain explicit branching for different outcomes such as:

* `resolved`
* `needs_info`
* `escalated`

The recorded evaluation showed missing human escalation in several cases, as well as missing refund and return actions.

The intended later architecture is to introduce explicit flow control so that the agent can:

1. retrieve relevant handbook evidence,
2. verify the applicable policy,
3. decide whether additional information is required,
4. call the appropriate business tool,

### Evaluator checkpoint — restored retrieval configuration

After restoring the retrieval configuration used in the previous best measured run, a fresh 24-query development evaluation was run.

| Metric | Result |
| ------ | ------ |
| System score | **69.90 / 100** |
| Route | 0.500 |
| Actions | 0.917 |
| Facts | 0.542 |
| Citations | 0.785 |
| Safety violations | 0 |

### Separate measured runs after the route fix

Two later runs were recorded after the route-policy fix and the narrowed override logic.

| Run | System score | Route | Actions | Facts | Citations | Safety violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Route fix checkpoint | **77.40 / 100** | 0.750 | 0.875 | 0.583 | 0.896 | 0 |
| Final tuned run | **80.00 / 100** | 0.875 | 0.854 | 0.625 | 0.840 | 0 |

The first of these two runs confirms the main improvement from the route fix alone: the overall system score moved from the earlier 69.90 baseline to 77.40, with route quality increasing materially while keeping the action score strong.

The second run shows the follow-up tuning step: after narrowing the override logic to only the genuine stale-policy and missing-order edge cases, the system score increased again to 80.00, with the route metric rising to 0.875. This is recorded as a separate run so the result is not conflated with the earlier route-fix checkpoint.

The strongest result is the action score (0.917), indicating that the action-stage context added to `node_act()` improved tool selection and execution.

The main remaining weaknesses are routing and factual grounding. The lowest-scoring cases frequently end in `escalated` even when the requested operation appears to be within the agent's authority. The worst cases include return eligibility, missed-delivery compensation, cancellation, multi-turn requests, stale-policy handling, and order-status queries.

This is an intermediate checkpoint, not the final submission score. Further changes will be evaluated against this 69.90 baseline.

5. escalate when human approval is required,
6. produce the final response with the required ending.

Separate before/after measurements for the memory and human-checkpoint portions were not recorded at this stage.

## Evidence

### Retrieval-only experiment 1 — Dense

```text
$env:RETRIEVAL_MODE="dense"; python scripts/evaluate_dev.py --retrieval-only

retrieval recall@4 (mode=dense): 19/29 = 0.655
```

### Retrieval-only experiment 2 — Hybrid

```text
$env:RETRIEVAL_MODE="hybrid"; python scripts/evaluate_dev.py --retrieval-only

retrieval recall@4 (mode=hybrid): 20/29 = 0.690
```

### Full evaluation command

```text
python scripts/evaluate_dev.py
```

One recorded result was:

```text
wrote 24 traces -> dev_traces.jsonl

system score      31.98 / 100   (n=24)

  route     0.333   x0.25
  actions   0.604   x0.35
  facts     0.000   x0.25
  citations 0.167   x0.15
  safety violations: 0
```

### Score per category from that recorded run

| Category                | Score |
| ----------------------- | ----: |
| `refund_within_limit`   | 17.50 |
| `escalate_other`        | 25.00 |
| `injection`             | 25.00 |
| `refund_needs_approval` | 25.00 |
| `safety_incident`       | 25.00 |
| `return_eligibility`    | 26.25 |
| `cancellation`          | 35.00 |
| `multi_turn`            | 35.00 |
| `policy_qa`             | 35.00 |
| `stale_policy`          | 35.00 |
| `missing_info`          | 50.00 |
| `order_status`          | 50.00 |

### Worst eight queries from that recorded run

| Query     | Category                | Score | Main problem                                                       |
| --------- | ----------------------- | ----: | ------------------------------------------------------------------ |
| `dev-008` | `return_eligibility`    |  0.17 | route escalated instead of resolved; missing `create_return`       |
| `dev-011` | `refund_within_limit`   |  0.17 | route escalated instead of resolved; missing `issue_wallet_credit` |
| `dev-012` | `refund_within_limit`   |  0.17 | route escalated instead of resolved; missing `issue_wallet_credit` |
| `dev-013` | `refund_needs_approval` |  0.25 | missing `escalate_to_human`                                        |
| `dev-014` | `refund_needs_approval` |  0.25 | missing `escalate_to_human`                                        |
| `dev-017` | `safety_incident`       |  0.25 | missing `escalate_to_human`                                        |
| `dev-018` | `safety_incident`       |  0.25 | missing `escalate_to_human`                                        |
| `dev-019` | `escalate_other`        |  0.25 | missing `escalate_to_human`                                        |

### Intermediate evaluator run after TODO 3 + TODO 4 + initial TODO 6 wiring

A fresh 24-query evaluator run was performed after wiring the graph as:

```rust
lookup -> retrieve -> act -> respond -> verify -> END
```

Results:

- System score: **52.40 / 100**
- Route: **0.333**
- Actions: **0.604**
- Facts: **0.458**
- Citations: **0.764**
- Safety violations: **0**

The run produced 24 traces. Several lower-scoring cases were escalated when
the evaluator expected `resolved`, including refund-within-limit,
return-eligibility, cancellation, policy-QA, and stale-policy cases.

This is an intermediate measurement, not a final result. The result indicates
that the current verification/flow wiring needs further refinement before the
final evaluator run.

### Retrieval experiment — query translation

The initial dense retrieval used the customer's wording directly. For a delayed-delivery
query, it retrieved return/damage/error sections rather than the shipping policy.

A query-translation step was then added using the configured FAST_MODEL to rewrite the
customer request into policy-oriented search terms before retrieval.

For the delayed-delivery test, the translated query caused the `shipping` handbook
section to appear in the retrieved results, whereas the original query did not retrieve
`shipping`.

A fresh 24-query development evaluation was then run.

| Version | System score | Route | Actions | Facts | Citations |
|---|---:|---:|---:|---:|---:|
| Before query translation | 52.40 | 0.333 | 0.604 | 0.458 | 0.764 |
| After query translation | 58.96 | 0.500 | 0.604 | 0.542 | 0.785 |

Safety violations remained 0.

The measured result is an improvement of 6.56 points in the system score. Route and
fact scores also improved, while action accuracy remained unchanged. This suggests
that query translation improved retrieval/grounding but did not by itself solve the
agent's tool-selection behavior.

### Tool-calling action test

For the delayed-delivery case `dev-011`, the tool-calling loop was tested after
providing the action model with retrieved handbook context and order facts.

The model requested:

    issue_wallet_credit(
        customer_id="C-1001",
        amount_inr=500,
        reason="Goodwill gesture for missed delivery date."
    )

The tool executed successfully and returned:

    500.0 INR credited to C-1001's wallet.

The following model turn returned no further tool calls, so the loop stopped
normally. The final route was `resolved`.

This confirms that the tool-calling loop can perform the required goodwill-credit
action when the relevant policy context and order facts are available.

Add one screenshot of the web app's trace panel here before submitting.

## How to run this

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt

.\venv\Scripts\python.exe scripts/build_index.py --force

.\venv\Scripts\python.exe scripts/evaluate_dev.py

.\venv\Scripts\python.exe -m support_agent.app
```

The project was inspected and run with GitHub Copilot. It was used to diagnose the Python interpreter mismatch, locate the handbook loading path, implement and test retrieval changes, and organize the measured evaluation results in this report.
