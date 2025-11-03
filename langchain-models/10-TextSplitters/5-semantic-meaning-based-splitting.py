from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
    )

text="""
Farmers were working hard in the fields, tending to their crops and ensuring a good harvest. The sun was shining brightly, and the birds were chirping in the trees. It was a beautiful day in the countryside.
The Indian Premier League (IPL) is one of the most popular cricket leagues in the world. It features top players from around the globe competing in a fast-paced and exciting format. The league has a massive following and is known for its thrilling matches and high-energy atmosphere.

Terrorism is a global issue that affects many countries. It involves the use of violence and intimidation to achieve political or ideological goals. Governments and organizations around the world are working
"""

docs=text_splitter.create_documents([text])
print(len(docs))
print(docs[1])
