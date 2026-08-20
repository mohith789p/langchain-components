from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# Relevant health & wellness documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

vector_store = FAISS.from_documents(
    documents = docs,
    embedding = embedding_model
)
similairy_retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 5})
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {"k" : 5}),
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
)

query = "How to improve energy levels and maintain balance?"

s_result = similairy_retriever.invoke(query)

mq_result = multiquery_retriever.invoke(query)

for i, doc in enumerate(s_result):
    print(f"Result {i + 1}")
    print(doc.page_content)

print("=" * 75)

for i, doc in enumerate(mq_result):
    print(f"Result {i + 1}")
    print(doc.page_content)