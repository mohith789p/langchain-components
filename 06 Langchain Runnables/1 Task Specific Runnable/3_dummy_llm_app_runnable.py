from abc import ABC, abstractmethod
import warnings

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
        return {"response" : "Here is the response for {}".format(prompt)}
    
    def predict(self, prompt):
        warnings.warn("Use invoke instead of predict, it is deprecated after 2026", DeprecationWarning)
        return {"response" : "Here is the response for {}".format(prompt)}

class DummyPromptTemplate(DummyRunnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_data):
        return self.template.format(**input_data)

    def format(self, input_data):
        warnings.warn("Use invoke instead of format, it is deprecated after 2026", DeprecationWarning)
        return self.template.format(**input_data)

class DummyStrOutputParser(DummyRunnable):
    def __init__(self):
        pass

    def invoke(self, result):
        return result["response"]

    def parse(self, result):
        warnings.warn("Use invoke instead of parse, it is deprecated after 2026", DeprecationWarning)
        return result["response"]


llm = DummyLLM(model = "gemini-2.5-flash")

template = DummyPromptTemplate(
    template = "Write a {length} poem on {topic}",
    input_variables = ["length", "topic"]
)

parser = DummyStrOutputParser()

chain = DummyRunnableConnector([template , llm , parser])

response = chain.invoke({
    "length" : "short", 
    "topic" : "Chess"
    })

print(response)

#  Uncomment belows lines to see the warnings
 
# prompt = template.format({
#     "length" : "short", 
#     "topic" : "Chess"
#     })

# response = llm.predict(prompt)

# result = parser.parse(response)

# print(result)