from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./8 Langchain Text Splitters/data/GA_CS_2026_Syllabus.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10,
    separator = ''
)

result = splitter.split_documents(docs)

for i in range(len(result)):
    print(i, ":", result[i].page_content)