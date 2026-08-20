from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name = "history"),
    ("human" , "{query}")
]) # available roles 'human', 'user', 'ai', 'assistant', or 'system'.

chat_history = []

with open("./2 Langchain Prompts/9_message_placeholder.txt") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    "history" : chat_history,
    "query" : "It's been one week, Where is my refund?"
})

print(prompt)