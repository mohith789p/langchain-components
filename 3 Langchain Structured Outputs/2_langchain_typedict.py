from typing import TypedDict, Optional, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

class Review(TypedDict):
    name : Annotated[Optional[str], "Return the actual reviewer's name only if explicitly mentioned in the review. Never infer or generate a name. Return None if the reviewer name is not explicitly provided."]
    key_themes : Annotated[str, "Write down all the key themes discuss in the review"]
    summary : Annotated[str, "A brief summary of the review under 2 to 3 sentences"]
    sentiment : Annotated[Literal["pos", "neg"], "Return the sentiment of the review in either positive or negative"]
    pros : Annotated[list[str], "Write down all the pros in a list if exist; otherwise return None"]
    cons : Annotated[list[str], "Write down all the cons in a list if exist; otherwise return None"]

structured_model = model.with_structured_output(Review)

# print(structured_model)

review_text = "I really love playing this game,buh I prefer selecting 3 stars for it because this game really luck some potentials,I suggest there should be additional stuffs like curves, runabouts,parking lots,roads signs for high and low lands and even traffic lights,all these will constitute to make the game more remarkable, exceptional and satisfactionary,Moreover, something like a competition with local players in Bluetooth or wifi connection will also do it good and make this more challenging."

result = structured_model.invoke(review_text)

print(result)
print(type(result))
