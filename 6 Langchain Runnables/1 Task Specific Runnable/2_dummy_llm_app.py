class DummyLLM:
    def __init__(self, model = "gemini-5-pro"):
        self.model = model

    def predict(self, prompt):
        return {"response" : "Here is the response for {}".format(prompt)}

class DummyPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_data):
        return self.template.format(**input_data)

class DummyStrOutputParser:
    def __init__(self):
        pass

    def parse(self, result):
        return result["response"]


llm = DummyLLM(model = "gemini-2.5-flash")

template = DummyPromptTemplate(
    template = "Write a {length} poem on {topic}",
    input_variables = ["length", "topic"]
)

parser = DummyStrOutputParser()

prompt = template.format({
    "length" : "short", 
    "topic" : "Chess"
    })

response = llm.predict(prompt)

result = parser.parse(response)

print(result)

# No standard interface, so components cannot form a chain