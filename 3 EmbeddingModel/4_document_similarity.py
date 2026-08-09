from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "./models"
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Machine Learning enables systems to learn patterns from data and make predictions or decisions without explicit programming.",
    "Deep Learning uses multi-layered neural networks to learn complex patterns from large amounts of data.",
    "Gen AI creates new content such as text, images, audio, or code by learning patterns from existing data."
]

query1 = "What is Machine Learning?"
query2 = "Explain Gen AI."

doc_embeddings = embedding.embed_documents(documents)
query1_embedding = embedding.embed_query(query1)

result = cosine_similarity([query1_embedding], doc_embeddings)[0]
index, score = sorted(enumerate(result), key = lambda x : x[1])[-1]

print("Query:", query1)
print(f"Matched Document: {documents[index]} (score: {score})")

query2_embedding = embedding.embed_query(query2)

result = cosine_similarity([query2_embedding], doc_embeddings)[0]
index, score = sorted(enumerate(result), key = lambda x : x[1])[-1]

print("Query:", query2)
print(f"Matched Document: {documents[index]} (score: {score})")