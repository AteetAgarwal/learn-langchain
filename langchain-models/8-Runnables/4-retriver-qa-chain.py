from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

#Load the document
file_path = Path(__file__).parent / "docs.txt"
loader = TextLoader(str(file_path))
documents = loader.load()

#Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

#Create embeddings
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

#Convert text into embeddings and store in vector database
vector_store = FAISS.from_documents(docs, embeddings)

#Create a retriever
retriever = vector_store.as_retriever()

#initialize the language model
model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

#create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

query = "Explain the concept of langchain."
answer = qa_chain.run(query)

print(f"Answer: {answer}")
