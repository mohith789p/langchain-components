from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# chat_template = ChatPromptTemplate([
#     SystemMessage(content = "You are a expert in {domain} and you can explain with simple examples connecting the technical terms."),
#     HumanMessage(content = "Explain the internal architecture of {topic}.")
# ])

chat_template = ChatPromptTemplate([
    ("system", "You are a expert in {domain} and you can explain with simple examples connecting the technical terms."),
    ("human", "Explain the internal architecture of {topic}.")
]) # available roles 'human', 'user', 'ai', 'assistant', or 'system'.


prompt = chat_template.invoke({
    "domain": "Edge Computing",
    "topic" : "Jetson Nano edge device"
})

print(prompt)