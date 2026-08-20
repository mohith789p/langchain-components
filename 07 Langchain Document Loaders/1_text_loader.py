from langchain_community.document_loaders import TextLoader

loader = TextLoader("7 Langchain Document Loaders/data/sample.txt")
docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))

print("Page Content:", docs[0].page_content)
print("Meta Data:", docs[0].metadata)