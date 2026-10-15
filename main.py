from dotenv import load_dotenv
from typing import Literal
load_dotenv()


from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools


MAX_ITERATIONS = 2  # Maximum number of iterations for the graph execution


# Define nodes and edges for the state graph
def draft_node(state:MessagesState):
    """Draft node to generate the initial answer."""

    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]} 


def revise_node(state:MessagesState):
    """Revise the answer based on tool results."""

    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:          # Literal is used to restrict the retun type to either "execute_tools" or END.
    """Determine whether to continue or end based on iteration count."""

    # Count number of tool calls in the messages
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    
    # The number of tool calls is the same as the number of iterations.
    num_iterations = count_tool_visits

    # Implement the conditional logic to determine whether to continue or end the loop.
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# Nodes
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

# Edges
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

# Visualize the graph structure
#print(graph.get_graph().draw_mermaid())  # Visualize the graph structure using Mermaid syntax
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")            # Draw the state graph PNG


res = graph.invoke({
    "messages": [{
        "role": "user",
        "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
    }]
})


# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]


if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])         # Print the output of the last tool call
print(res)