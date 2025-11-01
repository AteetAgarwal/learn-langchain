from langchain.llms import AzureChatOpenAI
#from langchain_community.llms.openai import AzureOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()
llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")

prompt = PromptTemplate(
    template="Suggest a catchy title for the following blog post about {topic}.",
    input_variables=["topic"]
)

topic = input("Enter the blog post topic: ")

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print(f"Suggested Blog Title: {blog_title}")