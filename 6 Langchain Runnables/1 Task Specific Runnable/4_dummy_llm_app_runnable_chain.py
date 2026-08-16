from abc import ABC, abstractmethod

class DummyRunnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

class DummyRunnableConnector(DummyRunnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

class DummyLLM(DummyRunnable):
    def __init__(self, model = "gemini-5-pro"):
        self.model = model

    def invoke(self, prompt):
        return {"response" : "Here is the response for \"{}\" ".format(prompt)}
    
    def predict(self, prompt):
        return {"response" : "Here is the response for \"{}\" ".format(prompt)}

class DummyPromptTemplate(DummyRunnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_data):
        return self.template.format(**input_data)

    def format(self, input_data):
        return self.template.format(**input_data)

class DummyStrOutputParser(DummyRunnable):
    def __init__(self):
        pass

    def invoke(self, result):
        return result["response"]

    def parse(self, result):
        return result["response"]


llm = DummyLLM(model = "gemini-2.5-flash")

template1 = DummyPromptTemplate(
    template = "Generate detailed notes on {topic}",
    input_variables = ["topic"]
)

template2 = DummyPromptTemplate(
    template = "Give me a summary on the {response}",
    input_variables = ["response"]
)

parser = DummyStrOutputParser()

chain1 = DummyRunnableConnector([template1, llm])
chain2 = DummyRunnableConnector([template2, llm , parser])

chain = DummyRunnableConnector([chain1, chain2])

response = chain.invoke({
    "topic" : "Chess"
    })

print(response)