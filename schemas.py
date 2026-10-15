from typing import List
from pydantic import BaseModel, Field           # Field is pydantic object to provide metadata inside a model.


# Class to define the structure of the reflection (critique)
# This class will be used alongside funciton call to provide structured output for the reflection agent to fill up these values.
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")
    # superfluous meaning: Unnecessary information and information that deos not add value to the answer.


# Class to define the sctructure of the answer to the question.
class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
        # This is a trick to prompt the LLM through class description.
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")


# Define a RevisedAnswer class that inherits from AnswerQuestion and adds a references field.
class RevisedAnswer(AnswerQuestion):            # Inherit from AnswerQuestion class to reuse the answer, reflection, and search_queries fields.
    """Revise your original answer to your question."""

    # Add refrences field
    references: List[str] = Field(description="Citations motivating your updated answer.")