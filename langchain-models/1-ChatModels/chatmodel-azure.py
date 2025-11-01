from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

result=model.invoke("Tell me a joke about computers.")
print(result)