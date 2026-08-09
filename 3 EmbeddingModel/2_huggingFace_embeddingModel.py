from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id = "sentence-transformers/all-MiniLM-L6-v2",
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

print(vectors)
print("Dim:", len(vector))