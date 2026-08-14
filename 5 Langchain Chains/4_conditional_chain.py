from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt1 = PromptTemplate(
    template="Write an appropriate response to this positive feedback in one paragraph \n {feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this negative feedback in one paragraph \n {feedback}",
    input_variables=["feedback"]
)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["pos", "neg"] = Field(description="Give the sentiment of the given feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt3 = PromptTemplate(
    template="Classify the feedback into sentiment such as positive or negative from the following feedback \n {feedback} \n {format}",
    input_variables=["feedback"],
    partial_variables={"format": parser2.get_format_instructions()}
)

classify_chain = prompt3 | model | parser2

chain1 = prompt1 | model | parser1
chain2 = prompt2 | model | parser1

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "pos", chain1),
    (lambda x: x.sentiment == "neg", chain2),
    RunnableLambda(lambda x: "couldn't find sentiment")
)

chain = classify_chain | branch_chain

# result = chain.invoke({"feedback": "This is a beautiful phone"})
result = chain.invoke({"feedback": "This is a terrible phone"})

print(result)