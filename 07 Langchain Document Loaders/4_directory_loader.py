from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "7 Langchain Document Loaders/data",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()    

print(type(docs))
print(len(docs))

for doc in docs:
    print(doc.metadata)


# limitation:
# it takes time to load if we give 500 files at once.