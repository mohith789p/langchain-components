from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="Google Genai provides powerful embedding models."),
]
vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    collection_name = "sample"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "what is chroma used for?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"Result {i + 1}")
    print(doc.page_content)