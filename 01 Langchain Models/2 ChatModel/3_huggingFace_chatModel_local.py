from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

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

response = model.invoke("Explain NVIDIA in 3 bullet points.")

print(response.content)