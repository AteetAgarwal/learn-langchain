from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()   
model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant."),
    ('human', "Explain in simple term, what is {topic}")
])

prompt = chat_template.invoke({'domain': 'medical', 'topic': 'quantum computing'})

print(prompt)
