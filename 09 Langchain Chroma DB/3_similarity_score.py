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

result = vector_store.similarity_search_with_score(
    query = "Who among these are bowlers?",
    k = 2
)

print(result)