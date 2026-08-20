from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature = 1.2
)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input in ["exit", "q", "quit"]:
        print("AI: Have a great time! Feel free to return whenever you want to resume.")
        break

    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI:", result.content)

print("=" * 45)
print("Chat History:")
print(chat_history)
print("=" * 45)