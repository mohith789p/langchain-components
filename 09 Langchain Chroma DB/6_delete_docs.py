from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

vector_store = Chroma(
    collection_name = "sample",
    embedding_function = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001"),
    persist_directory = "9 Langchain Chroma DB/chroma_db"
)

# id may be change
vector_store.delete(
    ids = ["52f1c57a-c247-48d3-a9d1-72d94a805fdb"]
)

result = vector_store.get()

print(result['documents'])
print(len(result['documents']))