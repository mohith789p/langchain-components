from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

template1 = PromptTemplate(
    template = "Generate one humourous joke on {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = "Explan the following joke {text}",
    input_variables = ["text"]
)
seq_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({    
    "joke" : RunnablePassthrough(),
    "explain" : RunnableSequence(template2, model, parser)
})

chain = RunnableSequence(seq_chain, parallel_chain)

result = chain.invoke({"topic" : "Cricket"})

print("joke:", result['joke'])
print("explanation:", result['explain'])

# passthrough ensures that joke is preserved