# LangChain Models

This project demonstrates the usage of various LangChain integrations with different AI models and services.

## Requirements

- Python >= 3.10
- UV package manager
- Dependencies listed in `pyproject.toml`

## Environment Setup with UV Package Manager

### Step 1: Install UV Package Manager

#### Linux/macOS (using curl):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Linux/macOS (using wget):
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

#### Windows (using PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative: Install via pip (if you already have Python):
```bash
pip install uv
```

#### Alternative: Install via pipx:
```bash
pipx install uv
```

### Step 2: Verify UV Installation
```bash
uv --version
```

### Step 3: Project Setup with UV

#### Clone and navigate to the project:
```bash
git clone <your-repo-url>
cd langchain-models
```

#### One-command setup (recommended):
```bash
# This automatically creates a virtual environment and installs all dependencies from pyproject.toml
uv sync
```

**That's it!** No need to manually create or activate virtual environments. `uv sync` handles everything automatically.

### Step 4: Environment Variables Setup

Create a `.env` file in the root directory with your API keys:
```bash
# Create the .env file
touch .env
```

Add the following content to `.env`:
```
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

### Step 5: Run Examples

After setup, you can run any example:
```bash
uv run 1-ChatModels/chatmodel-hf.py
```

### Useful UV Commands

- **Add a new dependency:**
  ```bash
  uv add package-name
  ```

- **Add development dependencies:**
  ```bash
  uv add --dev pytest black flake8
  ```

- **Remove a dependency:**
  ```bash
  uv remove package-name
  ```

- **Update dependencies:**
  ```bash
  uv sync --upgrade
  ```

- **Show installed packages:**
  ```bash
  uv pip list
  ```

- **Run Python scripts with UV:**
  ```bash
  uv run your-script.py
  ```

- **Install from requirements.txt (if available):**
  ```bash
  uv pip install -r requirements.txt
  ```

## Key Dependencies

- langchain (>= 0.3.27)
- langchain-anthropic (>= 0.3.22)
- langchain-openai (>= 0.3.35)
- langchain-huggingface (>= 0.3.1)
- langchain-google-genai (>= 2.0.10)
- transformers (>= 4.57.1)
- sentence-transformers (>= 5.1.1)
- huggingface-hub (>= 0.35.3)

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

## Project Structure

```
langchain-models/
├── main.py                           # Main entry point
├── pyproject.toml                    # Project configuration and dependencies
├── README.md                         # This file
├── .env                             # Environment variables (create this)
├── uv.lock                          # UV lock file (auto-generated)
├── 1-ChatModels/                    # Chat model implementations
│   ├── chatmodel-azure.py          # Azure OpenAI chat integration
│   ├── chatmodel-hf-local.py       # Local HuggingFace model chat
│   └── chatmodel-hf.py             # HuggingFace Hub chat integration
├── 2-EmbeddingModels/              # Embedding model implementations
│   ├── embedding-azure.py          # Azure OpenAI embeddings
│   └── embedding-local.py          # Local embedding models
├── 3-Document Similarity/          # Document similarity examples
│   └── document-similarity.py      # Document similarity using embeddings
├── 4-Prompts/                      # Prompt engineering examples
│   ├── 1-prompt-generator.py       # Basic prompt generation
│   ├── 2-prompt-ui.py             # Interactive prompt UI
│   ├── 3-prompt-ui-chain.py       # Chained prompt UI
│   ├── 4-messages.py              # Message handling
│   ├── 5-chatbot.py               # Simple chatbot implementation
│   ├── 6-chat-prompt-template.py  # Chat prompt templates
│   ├── 7-message-placeholder.py   # Message placeholders
│   ├── chat-history.txt           # Sample chat history
│   └── research_paper_summary_template.json # JSON template
├── 5-StructuredOutputs/            # Structured output examples
│   ├── 1-typeddictionary-demo.py   # TypedDict demonstrations
│   ├── 2-with-structured-output-typeddic.py # Structured TypedDict outputs
│   ├── 3-pydantic-demo.py         # Pydantic model examples
│   ├── 4-with-structured-output-pydantic.py # Structured Pydantic outputs
│   ├── 5-with-structured-output-json.py # JSON structured outputs
│   └── json-schema.json           # JSON schema definitions
├── 6-OutputParsers/               # Output parsing examples
│   ├── 1-string-output-parser-without-parser.py # Raw string parsing
│   ├── 2-string-output-parser.py  # String output parser
│   ├── 3-json-output-parser.py    # JSON output parser
│   ├── 4-structured-output-parser.py # Structured output parser
│   └── 5-pydantic-output-parser.py # Pydantic output parser
├── 7-Chains/                      # Chain implementations
│   ├── 1-simple-chain.py          # Basic chain example
│   ├── 2-sequential-chain.py      # Sequential chain processing
│   ├── 3-parallel-chain.py        # Parallel chain execution
│   └── 4-conditional-chain.py     # Conditional chain logic
├── 8-Runnables/                   # Runnable interface examples
│   ├── 1-simple-llm.py            # Simple LLM runnable
│   ├── 2-pdf-reader.py            # PDF reading runnable
│   ├── 3-llm-chains.py            # LLM chain runnables
│   ├── 4-retriver-qa-chain.py     # Retriever QA chain
│   ├── 5-problem-with-llm-classes-without-runnables.py # Legacy approach
│   ├── 6-standardized-llm-classes-mimic-runnables.py # Standardized approach
│   ├── 7-runnable-sequence.py     # Sequential runnables
│   ├── 8-runnable-parallel.py     # Parallel runnables
│   ├── 9-runnable-passthrough.py  # Passthrough runnables
│   ├── 10-runnable-lambda.py      # Lambda runnables
│   ├── 11-runnable-branch.py      # Branching runnables
│   └── docs.txt                   # Documentation
├── 9-DocumentLoaders/             # Document loading examples
│   ├── 1-text-loader.py           # Text file loader
│   ├── 2-pypdf-loader.py          # PDF loader using pypdf
│   ├── 3-directory-loader.py      # Directory loader
│   └── 4-directory-lazy-loader.py # Lazy directory loader
├── 10-TextSplitters/              # Text splitting strategies
│   ├── 1-length-based-text-splitting.py # Length-based splitting
│   ├── 2-text-structure-based-splitting.py # Structure-based splitting
│   ├── 3-document-based-language-splitting.py # Language-based splitting
│   ├── 4-document-based-markdown-splitting.py # Markdown splitting
│   └── 5-semantic-meaning-based-splitting.py # Semantic splitting
├── 11-Vectorstores/               # Vector store implementations
│   └── 1-vector-stores.py         # Vector store examples
├── 12-Retrievers/                 # Retrieval strategies
│   ├── 1-wikpedia-retrievers.py   # Wikipedia retrievers
│   ├── 2-vectorstore-retriver.py  # Vector store retrievers
│   ├── 3-maximum-marginal-reference-retriever-strategy.py # MMR strategy
│   ├── 4-multi-query-retriever-strategy.py # Multi-query strategy
│   └── 5-compression-contextual-retriever-strategy.py # Compression strategy
├── 13-RAG/                        # Retrieval-Augmented Generation
│   ├── 1-youtube-video-rag.py     # YouTube video RAG
│   └── 2-youtube-video-rag-chain.py # YouTube RAG chain
├── 14-AIAgents/                   # AI Agent implementations
├── 14-Tools/                      # Tool implementations
│   ├── 1-built-in-tools.py        # Built-in tools
│   ├── 2-custom-tools.py          # Custom tool creation
│   ├── 3-structured-tools.py      # Structured tools
│   ├── 4-basetools.py             # Base tool classes
│   ├── 5-toolkits.py              # Tool collections
│   ├── 6-tools-calling.py         # Tool calling examples
│   └── 7-currency-conversion-tools.py # Currency conversion tools
└── .venv/                         # Virtual environment (auto-created)
```
