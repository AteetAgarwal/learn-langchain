from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents=[
    "Delhi is the the capital of India",
    "Paris is the capital of the France",
    "Lucknow is the capital of Uttar Pradesh"
]
result=embedding.embed_documents(documents)
print(str(result))