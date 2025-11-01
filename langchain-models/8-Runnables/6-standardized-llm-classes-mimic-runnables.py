import random
from abc import ABC, abstractmethod


class Runnables(ABC):
    
    @abstractmethod
    def invoke(input_data):
        pass
    
    

class NakliLLM(Runnables):
    def __init__(self):
        print("NakliLLM initialized")
        
    def invoke(self, prompt: str):
        response_list=[
            "Delhi is the capital of India.",
            "IPL is a popular cricket league.",
        ]
        return {"response": random.choice(response_list)}

    def predict(self, prompt: str):
        response_list=[
            "Delhi is the capital of India.",
            "IPL is a popular cricket league.",
        ]
        return {"response": random.choice(response_list)}
    



class NakliPromptTemplate(Runnables):
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables
        print("NakloPromptTemplate initialized")
        
    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)
    


class NakliOutputStrParser(Runnables):
    def __init__(self):
        pass
    
    def invoke(self, input_data):
        return input_data['response']



class RunnableConnector(Runnables):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
        print("RunnableConnector initialized")

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data
    
    

template = NakliPromptTemplate(
    template="Suggest a catchy title for the following blog post about {topic}.",
    input_variables=["topic"]
)
llm = NakliLLM()
parser = NakliOutputStrParser()
chain = RunnableConnector([template, llm, parser])
print(chain.invoke({"topic": "LangChain"}))

#-----------------------------------------------------------------------------------------------------------
#Chaining multiple chain examples

template1 = NakliPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)
template2 = NakliPromptTemplate(
    template="Explain the following joke: {response}",
    input_variables=["response"]
)
llm1 = NakliLLM()
parser1 = NakliOutputStrParser()
chain1 = RunnableConnector([template1, llm1])
chain2 = RunnableConnector([template2, llm1, parser1])
final_chain = RunnableConnector([chain1, chain2])
final_chain.invoke({"topic": "LangChain"})