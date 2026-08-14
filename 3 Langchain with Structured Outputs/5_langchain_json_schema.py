from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

json_schema = {
  "title": "Review",
  "description": "Structured representation of a user review.",
  "type": "object",
  "properties": {
    "name": {
      "type": ["string", "null"],
      "description": "Name of the reviewer if explicitly mentioned; otherwise null."
    },
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "All key themes discussed in the review."
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review in 2 to 3 sentences."
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Overall sentiment of the review."
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "All explicitly mentioned advantages or positive aspects."
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "All explicitly mentioned disadvantages or negative aspects."
    }
  }
}

review_text = "I really love playing this game,buh I prefer selecting 3 stars for it because this game really luck some potentials,I suggest there should be additional stuffs like curves, runabouts,parking lots,roads signs for high and low lands and even traffic lights,all these will constitute to make the game more remarkable, exceptional and satisfactionary,Moreover, something like a competition with local players in Bluetooth or wifi connection will also do it good and make this more challenging."

structured_model = model.with_structured_output(json_schema)

# print(structured_model)

result = structured_model.invoke(review_text)

print(result)
print(type(result))
