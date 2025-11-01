from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()
model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

prompt = PromptTemplate(
    template="Suggest a catchy title for the following blog post about {topic}.",
    input_variables=["topic"]
)

chain = LLMChain(llm=model, prompt=prompt)

topic = input("Enter the blog post topic: ")
blog_title= chain.run(topic)

print(f"Suggested Blog Title: {blog_title}")