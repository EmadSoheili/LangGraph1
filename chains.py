from dotenv import load_dotenv
load_dotenv()

# from langchain_core.output_parsers.openai_tools import JSONOutputKeyToolsParser, PydanticOutputKeyToolsParser         # I am not using OpenAI tools
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser            # To parse the output of the chat model into a structured format
from langchain_core.messages import HumanMessage            # To create a human message for the chat model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder          # To create a prompt template for the chat model
from langchain_google_genai import ChatGoogleGenerativeAI          # To use the Google GenAI model for chat
from schemas import AnswerQuestion, RevisedAnswer
import datetime

MODEL = "gemini-3.5-flash-lite"


llm = ChatGoogleGenerativeAI(model=MODEL)          # To use the Google GenAI model for chat
parser = JsonOutputToolsParser(return_id=True)          # To parse the output of LLM into JSON format
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])           # To parse the output of LLM into AnswerQuestion Pydantic object


# Define the prompt template for the reflexion agent, which includes a list of messages.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answers the user's question above using the required format.")
    ]
).partial(time=datetime.datetime.now().isoformat(),)
            # {first_instruction} will be filled later with different prompts.


# Define the first responder prompt template
first_responder_prompt_template = actor_prompt_template.partial(            # partial is a LangChain method used to pre-fill some variables in a prompt template.
    first_instruction="Provide a detailed ~250 word answer.")
            

# Chain to initially answer the question and provide a critique of the answer.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion")           
            # In general model will decide to call a tool or not, and even it can output a simple string, but here we need a structured output.
            # tool_choice forces the LLM to use the AnswerQuestion tool to answer the question.
            # This guarantee we will get our defined structured output not a simple string.


revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


# Chain to revise the answer based on the critique and new information.
revisor = (actor_prompt_template.partial(first_instruction=revise_instructions) 
           | llm.bind_tools(tools=[RevisedAnswer], tool_choice="RevisedAnswer"))       
        

# This part is only for testing
if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital.")


    # This is a temporary test chian to test the first_responder chain and the output parser.
    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") 
        | parser_pydantic)


    res = chain.invoke(input={"messages": [human_message]})          # Invoke the chain with the human message as input
    print(res)