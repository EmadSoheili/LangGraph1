from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder
        # ChatPromptTemplate: will hold contents we will send or receive to and back from LLM
        # MessagesPlaceholder: will give us flexibility to put a placeholder for future messages we will get
from langchain_google_genai import ChatGoogleGenerativeAI


MODEL = "gemini-3.5-flash-lite"


# Create a prompt template for reflection
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),          # Placeholder for future messages we will get from the user
    ])


# Create a prompt template for generation
generation_prompt = ChatPromptTemplate.from_messages([
    (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
])


# Initialize LLM
llm = ChatGoogleGenerativeAI(model=MODEL)


# Create LangChain chains for generation and reflection
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm