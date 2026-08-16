from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

def word_counter(text):
    return len(text.split())

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

template1 = PromptTemplate(
    template = "Generate one humourous joke on {topic}. Output only the joke.",
    input_variables = ["topic"]
)

parser = StrOutputParser()

seq_chain = RunnableSequence(template1, model, parser)

# parallel_chain = RunnableParallel({    
#     "joke" : RunnablePassthrough(),
#     "count" : RunnableLambda(word_counter)
# })

parallel_chain = RunnableParallel({    
    "joke" : RunnablePassthrough(),
    "count" : RunnableLambda(lambda x : len(x.split()))
})

chain = RunnableSequence(seq_chain, parallel_chain)

result = chain.invoke({"topic" : "Cricket"})

print("joke:", result['joke'])
print("word count:", result['count'])