from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about the LangChain library.")
]

result = model.invoke(messages) 

messages.append(AIMessage(content=result.content))

print(messages)