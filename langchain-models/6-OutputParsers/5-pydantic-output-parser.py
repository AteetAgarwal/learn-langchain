import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()   

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The full name of the person")
    age: int = Field(gt=18,description="The age of the person in years")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

#1st prompt - Detailed Report
template= PromptTemplate(
    input_variables=["place"],
    template=("Generate the name, city, and age of a fictional {place} person.\n"
        "Return ONLY a JSON object that follows this structure:\n"
        "{format_instruction}"),
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result=chain.invoke({"place": "India"})
print(result)

#Atlternatively, you can do it step by step as shown below:
"""
prompt = template.invoke({"place": "India"})
print(prompt)

result= model.invoke(prompt)
print(result)

final_result = parser.parse(result.content)
print(final_result) 
print(type(final_result))
"""