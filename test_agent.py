from llm import GeminiLLM
from vector_db import VectorDatabase
from agent import EnterpriseRAGAgent


# ============================================================
# INITIALIZE
# ============================================================

llm = GeminiLLM()

db = VectorDatabase()


# ============================================================
# LOAD VECTOR DATABASE IF AVAILABLE
# ============================================================

if db.exists():
    db.load()


# ============================================================
# AGENT
# ============================================================

agent = EnterpriseRAGAgent(
    vector_db=db,
    llm=llm,
    documents=[],
    top_k=5,
    max_iterations=4
)


# ============================================================
# QUESTION
# ============================================================

query = input(
    "\nAsk Agent: "
).strip()


# ============================================================
# RUN
# ============================================================

result = agent.run(query)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(
    result["answer"]
)

print("\n" + "=" * 70)
print("AGENT TRACE")
print("=" * 70)

for step in result["trace"]:

    print(
        f"\nStep {step['step']}"
    )

    print(
        f"Action: {step['action']}"
    )

    print(
        f"Details: {step['details']}"
    )