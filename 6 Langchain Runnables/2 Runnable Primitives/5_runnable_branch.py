from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.7-flash")

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Summarized the following text. Do not exceed 300 words.\n {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

report_chain = RunnableSequence(template1, model, parser)
summary_chain = RunnableSequence(template2, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300, summary_chain),
    RunnablePassthrough()
)

chain = RunnableSequence(report_chain, branch_chain)

result = chain.invoke({"topic" : "Origin of Chess"})

print(result)