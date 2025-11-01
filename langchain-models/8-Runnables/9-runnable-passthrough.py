#Runnable pasthrough help to generate the same output as input had

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")
parser = StrOutputParser()

prompt1 =PromptTemplate(
    template="Generate a joke about {topic}.",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template="Generate the explanation about the response {topic}.",
    input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({"topic": "Narendra Modi"}))