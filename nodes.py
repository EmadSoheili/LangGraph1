from dotenv import load_dotenv
from langgraph.graph import MessagesState       # Keep track of Human and AI messages
from langgraph.prebuilt import ToolNode         # Execute tools by checking AI message to see if has a valid tool call
from react import llm, tools            # From react file we load llm and tools (list of tools)

load_dotenv()


SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

# Define the reasoning node
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """

    response = llm.invoke([{"role":"system", "content": SYSTEM_MESSAGE}, *state["messages"]])

    return {"messages": [response]}          # Append the AI message to the list of previous messages.


# Define the tool node
tool_node = ToolNode(tools)         # All items in tools are LangChain tools.