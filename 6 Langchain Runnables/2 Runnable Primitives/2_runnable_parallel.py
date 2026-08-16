from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

template1 = PromptTemplate(
    template = "Generate a single tweet about {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = "Generate a short linked post about {topic}",
    input_variables = ["topic"]
)

parallel_chain = RunnableParallel({
    "tweet" : RunnableSequence(template1, model, parser),
    "post" : RunnableSequence(template2, model, parser),
})

result = parallel_chain.invoke({"topic" : "AI"})

print("tweet:", result['tweet'])
print("post:", result['post'])
