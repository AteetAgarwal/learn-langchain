from langchain_community.document_loaders import CSVLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model= AzureChatOpenAI(deployment_name="gpt-4o-mini")

prompt = PromptTemplate(
    template="Answer the following question:\n{question} from the following text \n {text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()

loader = CSVLoader(file_path="social-network-ads.csv", encoding="utf-8")
documents = loader.load()
print(len(documents))

print(documents[0].page_content)
print(documents[0].metadata)

chain = prompt | model | parser
#print(chain.invoke({"text": documents[0].page_content, "question":"Who led the Series B round?"}))