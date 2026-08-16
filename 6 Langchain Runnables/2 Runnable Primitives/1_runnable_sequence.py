from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

template1 = PromptTemplate(
    template = "Generate one humourous joke on {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = "Explan the following joke {text}",
    input_variables = ["text"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({"topic" : "AI"})

print(result)