from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()   

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

#1st prompt - Detailed Report
template= PromptTemplate(
    input_variables=[],
    template="Give me the name, age, city of a fictional person \n: {format_instruction}",
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result=chain.invoke({})
print(result)

#Atlternatively, you can do it step by step as shown below:
"""
prompt = template.format()
print(prompt)

result= model.invoke(prompt)
print(result)

final_result = parser.parse(result.content)
print(final_result) 
print(type(final_result))
""" 