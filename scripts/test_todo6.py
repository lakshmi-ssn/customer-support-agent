import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support_agent.agent import SupportAgent

agent = SupportAgent(interrupt_before=["act"])

thread_id = "todo6-test-001"

query = {
    "query_id": "todo6-test-001",
    "query": "I want to return order MRD-700112",
    "customer_id": "C-1001",
    "history": [],
}

result = agent.resolve(query, thread_id=thread_id)

print("ROUTE:", result.get("route"))
print("ANSWER:", result.get("answer"))
print("STEPS:", result.get("steps"))
print("ACTIONS:", result.get("actions"))

print("\n--- RESUMING WITH APPROVAL ---")

result2 = agent.resume(
    thread_id=thread_id,
    approved=False,
    note="Approved by support supervisor."
)

print("ROUTE:", result2.get("route"))
print("ANSWER:", result2.get("answer"))
print("STEPS:", result2.get("steps"))
print("ACTIONS:", result2.get("actions"))