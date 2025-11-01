from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")
parser = StrOutputParser()

prompt1 =PromptTemplate(
    template="Generate a tweet about {topic}.",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template="Generate the linkedln post about {topic}.",
    input_variables=["topic"]
)


parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin   ": RunnableSequence(prompt2, model, parser)
})
print(parallel_chain.invoke({"topic": "Narendra Modi"}))