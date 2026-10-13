from dotenv import load_dotenv          # To load environment variables from .env file
load_dotenv()

from typing import TypedDict, Annotated         # To define a typed dictionary for structured data
                                                # TypedDict: to define a dictionary with hints for the keys
                                                # Annotated: to add metadata to the type hints
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph         # END: to define the end of a state graph
                                                    # StateGraph: to build stateful langgraph graphs
from langgraph.graph.message import add_messages            # Append messages to the state graph  
from chains import generation_chain, reflection_chain       # Import the generation and reflection chains from chains.py


class MessageGraph(TypedDict):
    """Schema for the messages in the state graph."""

    messages: Annotated[list[BaseMessage], add_messages]        # List of messages in the state graph, annotated with add_messages
    

REFLECT = "reflect"        # Constant for the reflection state
GENERATE = "generate"      # Constant for the generation state


def generation_node(state: MessageGraph):
    """Node for the generation state in the state graph."""

    # Invoke the generation chain with the history of messages
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}



def reflection_node(state: MessageGraph):
    """Node for the reflection state in the state graph."""

    # Append a temporary HumanMessage to satisfy Gemini's requirement 
    # that the conversation history must end with a user message.
    messages_for_gemini = state["messages"] + [HumanMessage(content="Please critique the tweet above and provide recommendations.")]

    # Invoke the reflection chain with the modified message list
    res = reflection_chain.invoke({"messages": messages_for_gemini})

    # Return the result of the reflection chain as a HumanMessage to feed into the next iteration of the state graph
    return {"messages": [HumanMessage(content=res.content)]}


# Build the state graph and tell LangGraph what is the state and how to update it
builder = StateGraph(state_schema=MessageGraph)


# Define nodes
builder.add_node(GENERATE, generation_node)          # Add the generation node to the state graph
builder.add_node(REFLECT, reflection_node)          # Add the reflection node to the state graph
builder.set_entry_point(GENERATE)                   # Set the entry point of the state graph to the generation node


# Define the function and the logic to be used in the conditional node
# The output is a string to indicate which node to go to next, either REFLECT or END
def should_continue(state: MessageGraph):
    """Determine whether to go to the reflect node or end node."""

    # Using a simple logic here instead of using LLM to decide whether to continue or not.
    if len(state["messages"]) >6:        # If the number of messages exceeds 6, end the state graph
        return END
    return REFLECT        # Otherwise, go to the reflect node


# Define edges
builder.add_conditional_edges(GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END})
            # Define the conditional edge from the generation and use the function should_continue to determine the next node, either REFLECT or END
builder.add_edge(REFLECT, GENERATE)        # Define the edge from the reflection node to the generation node


graph = builder.compile()        # Compile the state graph to be used in the main function

# Three ways to visualize the state graph, either in mermaid format, ASCII format, or PNG format:
#print(graph.get_graph().draw_mermaid())       # Print the state graph in mermaid format for visualization
#graph.get_graph().print_ascii()        # Print the state graph in ASCII format for visualization
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")            # Draw the state graph PNG


if __name__ == '__main__':
    print("Hello LangGraph!")
    
    # Define the input HumanMessage for the state graph 
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }


    response = graph.invoke(inputs)        # Invoke the state graph with the initial inputs

