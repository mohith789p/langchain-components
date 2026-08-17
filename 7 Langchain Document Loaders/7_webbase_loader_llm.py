from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

class Product(BaseModel):
    name: str = Field(description = "The name of the Product")
    company : str = Field(description = "Name of the company")
    provider : str = Field(description = "Name of the distributed")
    original_cost : float = Field(description = "Original Cost of the product") 
    final_cost : float = Field(description = "Cost of the product after discount") 
    specification : dict[str, str] = Field(description = "extract all the specification of the product in key-vale relation")

    def format_print(self):
        print("name:", self.name)
        print("company:", self.company)
        print("provider:", self.provider)
        print("cost (Original):", self.original_cost)
        print("cost (Final):", self.final_cost)
        print("Specifications:")
        for key, value in self.specification.items():
            print(f"\t {key} : {value}")

structured_model = model.with_structured_output(Product)

prompt = PromptTemplate(
    template = "Extract the details from the product {text}",
    input_variables = ["text"],
)

loader = WebBaseLoader(
    "https://www.web-scraping.dev/product/7" # using fake shopping due to lack of bot access in flipkart and amazon
)

docs = loader.load()

chain = prompt | structured_model

result = chain.invoke({'text': docs[0].page_content})

result.format_print()