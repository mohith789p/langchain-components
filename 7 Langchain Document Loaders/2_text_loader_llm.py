from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

prompt = PromptTemplate(
    template = "Generate a short Summary of the following text {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

loader = TextLoader("7 Langchain Document Loaders/data/sample.txt")

docs = loader.load()

chain = prompt | model | parser

response = chain.invoke({"text" : docs[0].page_content});

print("Summary:", response)