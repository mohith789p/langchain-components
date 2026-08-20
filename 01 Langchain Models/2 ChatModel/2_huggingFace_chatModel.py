from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("Explain NVIDIA in simple terms?")

print(response.content)