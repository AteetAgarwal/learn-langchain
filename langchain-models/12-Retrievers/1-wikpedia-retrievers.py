from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query="the geopolitical difference between india and pakistan from the perspective of China"

docs=retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:\n{doc.page_content}\n")
    print(f"Metadata: {doc.metadata}\n")
    
