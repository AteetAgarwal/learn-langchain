from langchain_core.tools import InjectedToolArg, tool
from typing import Annotated
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
  response = requests.get(url)
  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

llm = AzureChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion rate between USD and INR? And give INR values for 100 USD')]
ai_message = llm_with_tools.invoke(messages)
print(ai_message.tool_calls)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    tool_result = None
    if tool_call['name'] == 'get_conversion_factor':
        tool_result = get_conversion_factor.invoke(tool_call)
        conversion_rate=json.loads(tool_result.content)['conversion_rate']
        
    elif tool_call['name']  == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_result = convert.invoke(tool_call)
    messages.append(tool_result)
    
print(llm_with_tools.invoke(messages).content)

