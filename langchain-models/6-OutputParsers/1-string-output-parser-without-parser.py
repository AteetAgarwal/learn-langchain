from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()   

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

#1st prompt - Detailed Report
template1= PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on the topic: {topic}"
)

#2nd prompt - Summarize the report in 5 lines
template2= PromptTemplate(
    input_variables=["report"],
    template="Summarize the following report in 5 lines:\n\n{report}"
)

prompt1= template1.invoke({"topic": "black holes"})
result = model.invoke(prompt1)  

prompt2= template2.invoke({"report": result.content})
final_result = model.invoke(prompt2)  
print(final_result.content)