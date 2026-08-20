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

new_doc = Document(
    page_content="Virat Kohli is a master run-chaser and top-order batsman known for his high average, aggressive running between wickets, and exceptional consistency across all formats.",
    metadata={
        "team_name": "Royal Challengers Bengaluru",
        "team_code": "RCB",
    },
)

# id may be change
vector_store.update_document(
    document_id="d5f4d4f4-9284-424f-a666-c8efbd15a044", 
    document = new_doc

)

result = vector_store.get()

print(result['documents'])