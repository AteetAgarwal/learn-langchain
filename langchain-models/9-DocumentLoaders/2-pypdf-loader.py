from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader("GitHub Copilot - Presentation by vinitshahdeo.pdf")
documents = loader.load()
print(len(documents))

print(documents[0].page_content)
print(documents[1].metadata)

chain = prompt | model | parser
print(chain.invoke({"text": documents[0].page_content}))