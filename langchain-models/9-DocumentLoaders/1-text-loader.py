from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model= AzureChatOpenAI(deployment_name="gpt-4o-mini")

prompt = PromptTemplate(
    template="Summarize the following text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")
documents = loader.load()
print(documents)
print(type(documents))
print(len(documents))
print(type(documents[0]))
print(documents[0].page_content)

chain = prompt | model | parser
print(chain.invoke({"text": documents[0].page_content}))