from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
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