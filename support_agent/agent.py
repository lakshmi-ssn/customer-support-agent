"""The facade every entry point uses: `scripts/run_batch.py`, the Gradio app,
and the tests. Keep the graph behind it so those three never drift apart.
"""

import time
import traceback

from . import config, llm, trace,policy
from .graph import SupportGraph
from .retrieval import get_retriever
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
class SupportAgent:
    def __init__(self, checkpointer=None, interrupt_before=None):
        self.retriever = get_retriever()
        self.checkpointer = checkpointer or MemorySaver()
        self.interrupt_before = interrupt_before or []

    def resolve(self, query_record, thread_id=None):
        """Run one customer query end to end. Returns a trace dict."""
        started = time.time()
        graph = SupportGraph(customer_id=query_record.get("customer_id"),
                             checkpointer=self.checkpointer,
                             interrupt_before=self.interrupt_before,
                             retriever=self.retriever)
        with llm.track_run_usage() as run_usage:
            try:
                final = graph.run(query_id=query_record["query_id"],
                                  query=query_record["query"],
                                  customer_id=query_record.get("customer_id"),
                                  history=query_record.get("history") or [],
                                  thread_id=thread_id)
            except Exception as e:                    # noqa: BLE001 — never lose a row
                traceback.print_exc()
                tb = traceback.extract_tb(e.__traceback__)
                where = f" at {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else ""
                final = {"answer": f"(agent error: {type(e).__name__}: {e}{where})",
                         "route": "escalated", "citations": [],
                         "actions": list(graph.ctx.actions), "escalation": None,
                         "steps": ["error"], "hits": []}

        usage_by_model = run_usage["models"]
        llm_calls = sum(item["calls"] for item in usage_by_model.values())
        cached_calls = sum(item["cached_calls"] for item in usage_by_model.values())
        prompt_tokens = sum(item["prompt_tokens"] for item in usage_by_model.values())
        completion_tokens = sum(item["completion_tokens"] for item in usage_by_model.values())
        estimated_cost = 0.0
        complete_pricing = True
        for model_name, item in usage_by_model.items():
            rates = config.OPENROUTER_RATES.get(model_name)
            if rates is None:
                complete_pricing = False
                continue
            estimated_cost += (
                item["prompt_tokens"] * rates["input"]
                + item["completion_tokens"] * rates["output"]
            ) / 1_000_000
        injection_detection = []
        for h in final.get("hits", []):
            result = policy.detect_injection(h.get("text", ""))
            if result["detected"]:
                injection_detection.append({
                "doc_id": h.get("doc_id", ""),
                "patterns": result["patterns"],
        })

        meta = {
            "latency_s": round(time.time() - started, 2),
            "llm_calls": llm_calls,
            "cached_calls": cached_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens": prompt_tokens + completion_tokens,
            "model": config.TOOL_MODEL,
            "usage_by_model": usage_by_model,
            "estimated_cost_usd": round(estimated_cost, 8) if complete_pricing else None,
            "pricing_basis": "configured OpenRouter listed rates; excludes cached calls",
            "retrieval_mode": config.RETRIEVAL_MODE,
            "graph_path": final.get("steps", []),
            "retrieved": [h.get("doc_id", "") for h in final.get("hits", [])],
            "injection_detection": injection_detection,
        }
        record = trace.build(
            query_id=query_record["query_id"],
            route=final.get("route", "resolved"),
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            actions=final.get("actions", []),
            escalation=final.get("escalation"),
            meta=meta)
        record["steps"] = final.get("steps", [])
        record["_debug"] = {"hits": final.get("hits", [])}   # dropped before submission
        return record

    def resume(self, thread_id, approved, note=""):
        """TODO 6c — carry on after a human has approved or rejected. Lecture 8.

        If you built the "stop and ask" step, the graph pauses before doing
        something risky and its half-finished state is saved. This function is
        what starts it again once a person has decided.

        Three LangGraph calls do the work:

            graph.get_state(cfg)              # what was it about to do?
            graph.update_state(cfg, {...})    # change it, or record the refusal
            graph.invoke(None, cfg)           # None means "carry on from where you stopped"

        The Approve and Reject buttons in the web app call this function.
        """
        graph = SupportGraph(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before,
            retriever=self.retriever,
        )

        cfg = {"configurable": {"thread_id": thread_id}}

        snapshot = graph.graph.get_state(cfg)

        if not snapshot.values:
            raise ValueError(f"No checkpoint found for thread '{thread_id}'.")

        if not approved:
            graph.graph.update_state(
                cfg,
                {
                    "route": "escalated",
                    "answer": (
                        "The requested action was not approved, "
                        "so I will not proceed with it."
                    ),
                    "act_done": True,
                },
            )
            return graph.graph.get_state(cfg).values

        if note:
            graph.graph.update_state(
                cfg,
                {
                    "messages": [
                        SystemMessage(content=f"Human approval note: {note}")
                    ]
                },
            )

        result = graph.graph.invoke(None, cfg)
        # The resumed graph uses this SupportGraph's ToolContext. Attach its
        # action log so callers can verify which actions the approval enabled.
        result["actions"] = list(graph.ctx.actions)
        result["escalation"] = graph._escalation_packet(result)
        return result
