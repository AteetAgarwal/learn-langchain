from langchain.text_splitter import RecursiveCharacterTextSplitter


text="""LangChain is a framework for developing applications powered by language models. 
It provides a standard interface for all LLMs, as well as a collection of tools to work with them. 
It enables developers to build applications that can interact with documents, manage conversations, and integrate with various data sources. LangChain supports a variety of use cases, including chatbots, document analysis, and knowledge retrieval systems.
LangChain also offers seamless integration with popular vector databases like Pinecone, Chroma, and FAISS for efficient similarity search and retrieval. 
The framework includes built-in support for memory management, allowing applications to maintain context across multiple interactions. 
Additionally, LangChain provides agents that can use tools and make decisions, enabling more sophisticated AI workflows. 
It supports various document formats including PDF, CSV, HTML, and plain text, making it versatile for different data processing needs.
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks=text_splitter.split_text(text)
print(len(chunks))
print(chunks)
