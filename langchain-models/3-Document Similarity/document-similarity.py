from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

documents=[
    "India has won the ICC Cricket World Cup multiple times with strong team performances.",
    "Virat Kohli is known for his prolific run-scoring and consistency across formats.",
    "Spin bowlers often dominate on slow, turning pitches in subcontinental conditions.",
    "The Ashes is a historic Test series played between England and Australia.",
    "Twenty20 cricket popularized aggressive batting, power-hitting, and innovative shot-making."
]

query="Who is the leading run-scorer in international cricket?"

document_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], document_embeddings)[0]
index, score = sorted(list(enumerate(similarities)),key=lambda x:x[1])[-1]
print(documents[index])
print(f"Similarity Score: {score}")