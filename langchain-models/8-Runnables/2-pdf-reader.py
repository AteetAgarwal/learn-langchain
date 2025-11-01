from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS

load_dotenv()

#Load the document
loader = TextLoader('docs.txt')
documents = loader.load()

#Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

#Create embeddings
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")

#Convert text into embeddings and store in vector database
vector_store = FAISS.from_documents(docs, embeddings)

#Create a retriever
retriever = vector_store.as_retriever()

#Manually retrive relevant documents
query = "Explain the concept of vector databases." 
relevant_docs = retriever.get_relevant_documents(query)

#Combine the relevant documents into a single prompt
retrieved_text = "\n\n".join([doc.page_content for doc in relevant_docs])

#initialize the language model
model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

#Manually pass retrived text to the model
prompt = f"Using the following context, answer the question: {query} \n\n {retrieved_text}"
answer = model.predict(prompt)

print(f"Answer: {answer}")
