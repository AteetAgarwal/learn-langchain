from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "72XgD322wZ8"  # Example YouTube video ID

model = AzureChatOpenAI(deployment_name="gpt-4o-mini")
embedding_model = AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

try:
    # Create an instance
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id, languages=['en'])
    full_transcript = " ".join(item.text for item in transcript_data)
    print(full_transcript)
    
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    full_transcript = ""
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([full_transcript])
print(len(chunks))

vector_store = Chroma(
    embedding_function=AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
    persist_directory='my_chroma_db',
    collection_name='sample'
)
vector_store.add_documents(chunks)
#vector_store = FAISS.from_documents(chunks, embedding_model)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
prompt=PromptTemplate(
    input_variables=["context", "question"],
    template="""
            You are an AI assistant providing helpful answers to questions based on the provided context.
            Use the following pieces of context to answer the question at the end.  
            Answer only from the context.
            If the context does not contain the answer, say "I don't know".
            {context}
            Question: {question}
        """
)
question= "What is the main topic of the video?"
retrieved_docs=retriever.invoke(question)
context_text="\n\n".join([doc.page_content for doc in retrieved_docs])
final_prompt=prompt.invoke({"context": context_text, "question": question})

response=model.invoke(final_prompt)
print(response.content)