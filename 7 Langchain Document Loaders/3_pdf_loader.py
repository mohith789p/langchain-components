from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("7 Langchain Document Loaders/data/GA_CS_2026_Syllabus.pdf")
docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))

print("Page Content:", docs[0].page_content)
print("Meta Data:", docs[0].metadata)


# Types of pdf extractor

# Pdf with tables and columns => PDFPlumberLoader
# Scanned copies => AmazonTextractPDFLoader or UnstructuredPDFLoader
# need layout and image data => PyMuPDFLoader
# best structure extraction => UnstructuredPDFLoader