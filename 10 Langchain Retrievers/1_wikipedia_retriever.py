import os
from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever

load_dotenv() # for user_agent

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "Indian Premier League"

try:
    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"--- Result {i + 1} ---")
        print(f"Title: {doc.metadata.get('title')}")
        print(doc.page_content)
        print()

except Exception as e:
    print(f"✗ {query}")
    print(type(e).__name__, e)