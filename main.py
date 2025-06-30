from __future__ import annotations
import sqlite3, os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
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

@tool
def run_sql_tool(query: str) -> str:
    """Run a read-only SQL SELECT query on the internal BI database."""
    rows = run_sql(query)
    return str(rows)

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
    user_question = state["messages"][-1].content

    llm_with_tool = llm.bind_tools([run_sql_tool])

    system = {
        "role": "system",
        "content": f"""You are a SQL analyst assistant. Use the `run_sql_tool` tool to run SELECT queries.
        Only run SELECT queries. Schema is:

        {SCHEMA_DDL}

        If the user question requires SQL, call the tool with a valid SQL query.
        Otherwise, answer directly."""
    }

    result = llm_with_tool.invoke([
        system,
        {"role": "user", "content": user_question}
    ])

    print("🧠 Tool Call:", result.tool_calls)

    if result.tool_calls:
        tool_call = result.tool_calls[0]
        sql_result = run_sql_tool.invoke(tool_call["args"])
        # Let model explain the result
        explanation = llm.invoke([
            {"role": "system", "content": "Explain the query result to the user in natural language."},
            {"role": "user", "content": f"Question: {user_question}\nSQL Result: {sql_result}"}
        ])
        return {"messages": [{"role": "assistant", "content": explanation.content}]}
    else:
        return {"messages": [{"role": "assistant", "content": result.content or "No response."}]}

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
