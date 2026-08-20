from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.flipkart.com/u-s-polo-assn-full-sleeve-printed-men-sweatshirt/p/itmb105962bbad73?pid=SWSHHR8VMX3FQZU8&lid=LSTSWSHHR8VMX3FQZU8OV4QYW&marketplace=FLIPKART&q=hoddie+for+men%E2%80%99s&store=clo%2Fqvw%2F64a%2Fvui&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_na&fm=search-autosuggest")

docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))

print("Page Content:", docs[0].page_content)
print("Meta Data:", docs[0].metadata)