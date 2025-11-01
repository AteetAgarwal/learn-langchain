from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
documents=[
    "Delhi is the the capital of India",
    "Paris is the capital of the France",
    "Lucknow is the capital of Uttar Pradesh"
]
result=embedding.embed_documents(documents)
print(str(result))