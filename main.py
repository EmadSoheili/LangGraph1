from dotenv import load_dotenv
from langchain_core.messages import HumanMessage                # Input that we start the graph with
from langgraph.graph import MessagesState, StateGraph, END      # StateGraph creates the most generic graph
from nodes import run_agent_reasoning, tool_node

load_dotenv()


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1           # Index for the last message 


# If the last message is a tool call, then it will return ACT, else it will return End
def should_continue(state:MessagesState) -> str:
    last_message = state["messages"][LAST]
    if not getattr(last_message, "tool_calls", None):
        return END
    return ACT


# Define our nodes and edges in the graph
flow = StateGraph(MessagesState)

# Define nodes and entry point
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

# Defined which nodes will come after the agent reason node (Conditional Edge)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

# Define edge
flow.add_edge(ACT, AGENT_REASON)


app = flow.compile()            # Compile the graph

app.get_graph().draw_mermaid_png(output_file_path="flow.png")           # Output the graph as PNG


if __name__ == "__main__":
    print("Hello, ReAct LangGraph with Function Calling!")

    # Invoke the graph
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it.")]})
    print(res["messages"][LAST].text)

