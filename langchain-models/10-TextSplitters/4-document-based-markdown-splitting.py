from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text="""
# LangChain Models

This project demonstrates the usage of various LangChain integrations with different AI models and services.

## Project Structure

```
langchain-models/
├── ChatModels/                    # Chat model implementations
│   ├── chatmodel-azure.py        # Azure OpenAI chat integration
│   ├── chatmodel-hf-local.py     # Local HuggingFace model chat
│   └── chatmodel-hf.py           # HuggingFace Hub chat integration
├── EmbeddingModels/              # Embedding model implementations
│   ├── document-similarity.py     # Document similarity using embeddings
│   ├── embedding-azure.py        # Azure OpenAI embeddings
│   └── embedding-local.py        # Local embedding models
└── Document Similarity/          # Document similarity examples
```

## Requirements

- Python >= 3.10
- Dependencies listed in `pyproject.toml`

## Key Dependencies

- langchain (>= 0.3.27)
- langchain-anthropic (>= 0.3.22)
- langchain-openai (>= 0.3.35)
- langchain-huggingface (>= 0.3.1)
- langchain-google-genai (>= 2.0.10)
- transformers (>= 4.57.1)
- sentence-transformers (>= 5.1.1)
- huggingface-hub (>= 0.35.3)

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies using UV:
```bash
uv pip install -e .
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Features

### Chat Models
- Azure OpenAI integration
- HuggingFace models (both local and hub-based)
- Support for various LLM providers

### Embedding Models
- Azure OpenAI embeddings
- Local embedding models
- Document similarity analysis

## Usage Examples

### HuggingFace Chat Model
```python
from langchain_huggingface import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceHub(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

result = llm.predict("Tell me a joke about computers.")
print(result)
```

### Azure OpenAI Embeddings
```python
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Lucknow is the capital of Uttar Pradesh"
]
result = embedding.embed_documents(documents)
```

## Notes
- Make sure to have appropriate API keys set in your `.env` file
- For local models, ensure you have sufficient computational resources
- Some models may require additional dependencies (PyTorch, TensorFlow, etc.)

"""
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, 
    chunk_size=1000, 
    chunk_overlap=0
    )
chunks=text_splitter.split_text(text)
print(len(chunks))
print(chunks[0])
