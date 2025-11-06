from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

# tool create
@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))
print(multiply.name)
print(multiply.description)
print(multiply.args)

llm = AzureChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])
print(llm_with_tools.invoke('Hi, how are you?'))

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]
result = llm_with_tools.invoke(messages)
messages.append(result)
print(messages)

tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)
print(f"\nmessages: {messages}")
print(llm_with_tools.invoke(messages).content)