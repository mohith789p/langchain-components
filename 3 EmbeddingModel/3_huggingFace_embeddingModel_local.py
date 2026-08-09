from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "./models"
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

text = "I'm from India."
vector = embedding.embed_query(text)

print(vector)
print("\n" * 3)

documents = [
    "India is a diverse country with a rich cultural heritage.",
    "It is the world's most populous country.",
    "New Delhi is the capital of India."
]

vectors = embedding.embed_documents(documents)
print("Dim:", len(vector))