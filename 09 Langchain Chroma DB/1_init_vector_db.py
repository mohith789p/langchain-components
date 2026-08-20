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

docs = [
    Document(
        page_content="MS Dhoni serves as the primary wicket-keeper and finisher, known for his calm decision-making under high-pressure death overs and rapid stumpings.",
        metadata={
            "team_name": "Chennai Super Kings",
            "team_code": "CSK"
        }
    ),
    Document(
        page_content="Sunil Narine provides aggressive opening boundaries during powerplays and controls the middle overs with tight off-break mystery bowling.",
        metadata={
            "team_name": "Kolkata Knight Riders",
            "team_code": "KKR"
        }
    ),
    Document(
        page_content="Virat Kohli anchors the top order as a right-handed opening batsman, focusing on high strike rotation and chasing down target totals efficiently.",
        metadata={
            "team_name": "Royal Challengers Bengaluru",
            "team_code": "RCB"
        }
    ),
    Document(
        page_content="Jasprit Bumrah operates as the lead fast bowler, specializing in yorkers at the death and maintaining low economy rates across all powerplays.",
        metadata={
            "team_name": "Mumbai Indians",
            "team_code": "MI"
        }
    ),
    Document(
        page_content="Rinku Singh acts as a middle-order power hitter, specializing in clearing the boundaries during the final overs of run chases.",
        metadata={
            "team_name": "Kolkata Knight Riders",
            "team_code": "KKR"
        }
    )
]

vector_store.add_documents(docs)

result = vector_store.get(include=["embeddings", "documents"])
print(result)