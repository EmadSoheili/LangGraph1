from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_tavily import TavilySearch           # To search the internet
from langchain_core.tools import (StructuredTool)           # Allow us to convert a python function into a tool that can be called by the LLM
from langgraph.prebuilt import ToolNode         # Execute tools in a graph structure
from schemas import (AnswerQuestion, RevisedAnswer)         # Import the Pydantic models for structured output

tavily_tool = TavilySearch(max_results=5)  # Create a TavilySearch tool instance


# Define a tool to run the generated queries and return the results.
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""

    return tavily_tool.batch([{"query": query} for query in search_queries])
    # barch helps to run multiple queries in parallel.


# Created two tools using the run_queries function, one for AnswerQuestion and one for RevisedAnswer.
# The reason that we created two tools is that we want to have different tool names for the two different classes.
# It helps see different tool names in the graph or LangSmith when we are tracing.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=RevisedAnswer.__name__),
    ]
)
