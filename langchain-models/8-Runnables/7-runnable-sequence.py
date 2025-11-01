from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")
parser = StrOutputParser()

prompt1 =PromptTemplate(
    template="Generate a joke about {topic}.",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template="Explain the following joke {response}.",
    input_variables=["response"]
)


#chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

#Alternatively LCEL (Langchain Expression Language) can also be used for runnable sequence
chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({"topic": "Narendra Modi"}))