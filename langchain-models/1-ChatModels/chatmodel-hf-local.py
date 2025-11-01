import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = "C:\\D\\hf_cache"  # Set Hugging Face cache directory

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(max_new_tokens=100, temperature=0.7),
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("Tell me a joke about computers.")
print(result.content)       