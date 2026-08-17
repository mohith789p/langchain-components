from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("7 Langchain Document Loaders/data/github_incident_traffic_logs.csv")
docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))

print("Page Content:", docs[0].page_content)
print("Meta Data:", docs[0].metadata)