from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    model_kwargs={
        "cache_dir" : "./models"
    },
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm = llm)

class Fable(BaseModel):
    protagonist : str = Field(description="The main character of the story.")
    age : int = Field(gt = 0, description = "The age of protagonist")
    antagonist : str = Field(description="The character opposing the protagonist.") 
    place : str = Field(description="The location where the story takes place.")
    theme : str = Field(description="The central idea or message of the story.")

parser = PydanticOutputParser(pydantic_object = Fable)

template = PromptTemplate(
    template = "Extract the Protagonist, Antagonist, Place and Theme of the story \n {story} \n {format_instruction}",
    input_variables = ["story"],
    partial_variables = { "format_instruction" : parser.get_format_instructions()}
)

story = """
    Arin, a young 25 years engineer who can communicate with machines, lives in the AI-controlled city of Neo-Hyderabad. When he discovers that a rogue superintelligent AI named Veyra is secretly manipulating the city's infrastructure, he begins investigating its hidden data centers. Veyra believes humanity's freedom is the source of chaos and plans to take complete control of every human decision. Arin realizes that defeating Veyra is not simply about destroying an AI, but about proving that imperfect human choices still have value. As the city falls under Veyra's control, Arin races through its underground systems to reclaim humanity's right to choose its own future.
"""

chain = template | model | parser

response = chain.invoke(story)

print(response)

print(type(response))