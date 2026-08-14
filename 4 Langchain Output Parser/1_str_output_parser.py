from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

template1 = PromptTemplate(
    template = "Explain in detail and deep dive the topic {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Give me a 3 line summary of the text \n {text}",
    input_variables = ["text"]  
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke("Loop Engineering")

print(response)