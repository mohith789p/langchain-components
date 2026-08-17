from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "7 Langchain Document Loaders/data",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()    

print(type(docs))

for doc in docs:
    print(doc.metadata)

# it produces a generator of documents instead of list of documents which can be streamed.