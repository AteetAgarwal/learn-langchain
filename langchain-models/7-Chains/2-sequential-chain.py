from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")

prompt1 =PromptTemplate(
    template="Generate a detailed report about {topic}.",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template="Generate a 5 pointer summart for following text:\n{report}",
    input_variables=["report"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "cricket"})    
print(result)

chain.get_graph().print_ascii()