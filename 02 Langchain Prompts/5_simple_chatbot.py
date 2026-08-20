from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature = 1.2
)

while True:
    user_input = input("You: ")
    if user_input in ["exit", "q", "quit"]:
        print("AI: Have a great time! Feel free to return whenever you want to resume.")
        break

    result = model.invoke(user_input)
    print("AI:", result.content)