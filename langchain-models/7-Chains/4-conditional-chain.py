from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")

class SentimentClassification(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback text")

parser1 = PydanticOutputParser(pydantic_object=SentimentClassification)
parser2 = StrOutputParser()

prompt1 =PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}.",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser1.get_format_instructions()}
)

prompt2 =PromptTemplate(
    template="Write an appropriate response to this positive feedback.\n{feedback}",
    input_variables=["feedback"]
)

prompt3 =PromptTemplate(
     template="Write an appropriate response to this negative feedback.\n{feedback}",
    input_variables=["feedback"]
)

classifier_chain =prompt1 | model | parser1

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser2),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a awesome phone."})    

print(result)

chain.get_graph().print_ascii()