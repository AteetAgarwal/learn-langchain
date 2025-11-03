from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

#Way to split the text based on length
"""
text="LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs, as well as a collection of tools to work with them. It enables developers to build applications that can interact with documents, manage conversations, and integrate with various data sources. LangChain supports a variety of use cases, including chatbots, document analysis, and knowledge retrieval systems."
text_splitter = CharacterTextSplitter(separator="", chunk_size=100, chunk_overlap=20)
result=text_splitter.split_text(text)
"""

text_splitter = CharacterTextSplitter(separator="", chunk_size=200, chunk_overlap=40)
result=text_splitter.split_documents(docs)
print(result)