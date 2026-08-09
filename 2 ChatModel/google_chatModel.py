from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1.5)

messages = []

query1 = "I'm from South India."

messages.append(
    HumanMessage(query1)
)

response = model.invoke(messages)

print("Human :", query1)
print("AI:", response.content)

messages.append(response)

query2 = "List the top 5 spoken languages in my country."

messages.append(
    HumanMessage(query2)
)

response = model.invoke(messages)

print("Human :", query2)
print("AI:", response.content)

messages.append(response)

print("\n" * 4)

for message in messages:
    print(message)