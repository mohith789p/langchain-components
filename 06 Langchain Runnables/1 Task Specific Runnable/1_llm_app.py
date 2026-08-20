from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate(
    template = "Suggest a 5 catchy blog title about {topic}",
    input_variables = ["topic"]
)

prompt = template.format(topic = "Chess")

response = llm.predict(prompt)

print(response)