#Runnable pasthrough help to generate the same output as input had

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

def work_counter(text):
    return len(text.split())

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")
parser = StrOutputParser()

prompt1 =PromptTemplate(
    template="Write a detailed report about {topic}.",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template="Summarize the following text {text}.",
    input_variables=["text"]
)

report_gen_chain = RunnableSequence(prompt1, model, parser)
branch_chain=RunnableBranch(
    (lambda x: len(x.split())>100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain, branch_chain)
print(final_chain.invoke({"topic": "Russia vs Ukraine"}))