from __future__ import annotations
import sqlite3, os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ────────────────────────────────
# 0. Local model
# ────────────────────────────────
llm = ChatOllama(model="qwen2:7b", base_url="http://localhost:11434") # run it locally using ollama

# ────────────────────────────────
# 1. Fake BI DB
# ────────────────────────────────
DB = sqlite3.connect(":memory:")
cur = DB.cursor()
cur.executescript("""
CREATE TABLE sales(year INT, quarter INT, revenue NUM);
INSERT INTO sales VALUES
  (2024,1, 120000), (2024,2, 135000), (2024,3, 128000), (2024,4, 142000),
  (2025,1, 150000), (2025,2, 158000);

CREATE TABLE customers(year INT, quarter INT, new_customers INT);
INSERT INTO customers VALUES
  (2025,1, 2300), (2025,2, 2450);
""")
DB.commit()

def run_sql(query: str) -> list[tuple]:
    """Read-only SQL runner."""
    try:
        assert "delete" not in query.lower()
        rows = cur.execute(query).fetchall()
        return rows
    except Exception as e:
        return [("SQL-ERROR", str(e))]

# ────────────────────────────────
# 2. LangGraph state & helpers
# ────────────────────────────────
class Switch(BaseModel):
    message_type: Literal["sql", "text"]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify(state: State):
    user_msg = state["messages"][-1].content
    clf = llm.with_structured_output(Switch)
    tag = clf.invoke([
        {"role":"system","content":
         "Return 'sql' if the user asks for numbers, KPIs, summaries based on the database; "
         "otherwise 'text'."},
        {"role":"user","content": user_msg}
    ])
    return {"message_type": tag.message_type}

def router(state: State):
    return {"next": "sql_agent" if state["message_type"]=="sql" else "fallback"}

# ────────────────────────────────
# 3. SQL agent node
# ────────────────────────────────
SCHEMA_DDL = """
sales(year INT, quarter INT, revenue NUM)
customers(year INT, quarter INT, new_customers INT)
"""

def sql_agent(state: State):
    question = state["messages"][-1].content

    # 1) ask model for SQL
    sql = llm.invoke([
        {"role": "system", "content": f"""You are a SQL assistant. ONLY return a single SQLite SELECT query based on the following schema:

        {SCHEMA_DDL}

        Do not include explanations or anything else. Only return raw SQL without markdown or comments.
        """},
        {"role":"user","content":question}
    ]).content.strip("```sql\n").strip("```").strip()

    print(f"\n🧠 LLM-generated SQL:\n{sql}\n")  # ADD THIS

    rows = run_sql(sql)

    # 2) narrate
    explanation = llm.invoke([
        {"role":"system","content":"Answer the question using the rows strictly."},
        {"role":"user","content":f"Question: {question}\nRows: {rows}"}
    ])
    return {"messages":[{"role":"assistant","content":explanation.content}]}

# ────────────────────────────────
# 4. Fallback text agent
# ────────────────────────────────
def fallback(state: State):
    reply = llm.invoke(state["messages"])
    return {"messages":[{"role":"assistant","content":reply.content}]}

# ────────────────────────────────
# 5. Build & compile graph
# ────────────────────────────────
g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("router", router)
g.add_node("sql_agent", sql_agent)
g.add_node("fallback", fallback)

g.add_edge(START, "classify")
g.add_edge("classify", "router")
g.add_conditional_edges("router", lambda s: s["next"],
                        {"sql_agent":"sql_agent","fallback":"fallback"})
g.add_edge("sql_agent", END)
g.add_edge("fallback", END)

graph = g.compile()

# ────────────────────────────────
# 6. REPL loop
# ────────────────────────────────
def chat():
    state: State = {"messages": [], "message_type": None}
    while True:
        user = input("You: ")
        if user.lower() in {"exit","quit"}:
            break
        state["messages"] += [{"role":"user","content":user}]
        state = graph.invoke(state)
        print("Bot:", state["messages"][-1].content)

if __name__ == "__main__":
    chat()
