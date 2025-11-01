from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template="Summarize the following report in 5 lines only strictly:\n\n{report}"
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result= chain.invoke({"topic": "black holes"})
print(result)