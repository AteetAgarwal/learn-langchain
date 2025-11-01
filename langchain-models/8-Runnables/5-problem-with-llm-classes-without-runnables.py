import random


class NakliLLM:
    def __init__(self):
        print("NakliLLM initialized")

    def predict(self, prompt: str):
        response_list=[
            "Delhi is the capital of India.",
            "IPL is a popular cricket league.",
        ]
        return {"response": random.choice(response_list)}
    



class NakloPromptTemplate:
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables
        print("NakloPromptTemplate initialized")

    def format(self, input_dict):
        return self.template.format(**input_dict)
    



class NakliLLMChain:
    def __init__(self, llm: NakliLLM, prompt: NakloPromptTemplate):
        self.llm = llm
        self.prompt = prompt
        print("NakliLLMChain initialized")

    def run(self, input_dict):
        prompt_str = self.prompt.format(input_dict)
        result= self.llm.predict(prompt_str)
        return result["response"]
    
    

template = NakloPromptTemplate(
    template="Suggest a catchy title for the following blog post about {topic}.",
    input_variables=["topic"]
)
llm = NakliLLM()
chain = NakliLLMChain(llm=llm, prompt=template)
output=chain.run({"topic": "LangChain"})
print(f"Suggested Blog Title: {output}")