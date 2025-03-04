# RAG Application

## Overview
This RAG (Retrieval-Augmented Generation) application leverages advanced natural language processing techniques to enhance text generation by retrieving relevant information from a knowledge base. Built with LlamaIndex and integrating with multiple LLM providers, it provides a robust foundation for creating intelligent document retrieval and question-answering systems.

## Project Structure
```
RAG_Application/
├── src/
│ ├── components/
│ │ ├── LLM.py # LLM client implementations
│ │ ├── model.py # RAG model implementation
│ │ ├── prompts.py # Prompt templates and management
│ │ └── init.py # Package initialization
│ ├── logger.py # Logging configuration and utilities
│ ├── utils.py # Utility functions and helpers
│ └── init.py # Package initialization
├── logs/ # Directory for application logs
├── requirements.txt # Project dependencies
└── setup.py # Package setup configuration
└── setup.py          # Package setup configuration
```

## Features
- **Multi-Backend LLM Support**: 
  - Hugging Face API integration
  - Local model inference
  - OpenAI-compatible API support
- **Centralized Prompt Management**:
  - System prompts for different use cases
  - Query prompt templates
  - Easy customization and extension
- **Vector Storage & Retrieval**:
  - Integration with ChromaDB
  - Efficient similarity search
  - Persistent storage for embeddings
- **Robust Logging System**:
  - Comprehensive logging with timestamps
  - Multiple log levels
  - Automatic log directory management

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment management tool (optional but recommended)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG_Application
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create package structure:
   ```bash
   touch src/__init__.py
   touch src/components/__init__.py
   ```

## Configuration

### Environment Variables
Create a `.env` file:
```
HUGGINGFACE_API_KEY=your_hf_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional
CHROMA_DB_DIR=path/to/chromadb
LOG_LEVEL=INFO
```

## Usage

### 1. Basic RAG Model Setup
```python
from src.components.model import RAGModel

# Initialize RAG model
model = RAGModel(
    model_key="mistral-7b",
    backend="api",
    system_prompt_type="rag",
    query_prompt_type="default"
)

# Query the model
response = model.query("Your question here")
print(response)
```

### 2. Direct LLM Client Usage
```python
from src.components.LLM import get_llm_client

# Initialize LLM client
client = get_llm_client(
    model_key="mistral-7b",
    backend="api"
)

# Get completion
response = client.complete("Your prompt here")
print(response)
```

### 3. Working with Prompts
```python
from src.components.prompts import PromptTemplates

# Get system prompt
system_prompt = PromptTemplates.get_system_prompt("rag")

# Get query prompt
query_prompt = PromptTemplates.get_query_prompt("default")
```

## Supported Models

### Hugging Face Models
- `mistral-7b`: Mistral 7B Instruct
- `yi-6b`: Yi 6B Chat
- `phi-2`: Microsoft Phi-2
- `zephyr-7b`: Hugging Face Zephyr 7B

### OpenAI-Compatible Models
- `gpt-3.5-turbo`
- `gpt-4`
- `claude-3-opus`
- `claude-3-sonnet`

## Prompt Types

### System Prompts
- `rag`: For retrieval-augmented generation
- `chat`: For general conversation
- `summarization`: For document summarization
- `code`: For code-related tasks

### Query Prompts
- `default`: Standard query template
- Custom templates supported

## Error Handling

### Common Issues and Solutions

1. **Import Errors**:
   ```python
   import os, sys
   sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
   ```

2. **API Issues**:
   - Verify API keys in environment variables
   - Check model availability and access permissions
   - Ensure stable internet connection

3. **Prompt Errors**:
   - Validate prompt type existence
   - Check prompt template formatting
   - Ensure proper initialization of PromptTemplates

## Logging System

### Log Levels
- `INFO`: General operational messages
- `ERROR`: Error conditions
- `WARNING`: Warning messages
- `DEBUG`: Detailed debugging information

### Log Format
```
[timestamp] line_number name - level - message
```

## Performance Optimization
- Efficient document indexing strategies
- Optimized vector storage and retrieval
- Caching mechanisms for frequently accessed data

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## Troubleshooting
Common issues and their solutions:
1. **Installation Issues**:
   - Ensure Python version compatibility
   - Check virtual environment activation
   - Verify all dependencies are installed correctly

2. **Runtime Errors**:
   - Check log files for detailed error messages
   - Verify environment variables are set correctly
   - Ensure sufficient system resources

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- LlamaIndex team for the core retrieval framework
- LangChain community for the LLM tools
- ChromaDB team for the vector storage solution

## Contact
For any queries or support, please:
1. Open an issue in the repository
2. Contact the maintainers
3. Join our community discussions

---
**Note**: This project is under active development. Current implementation focuses on:
- Multi-backend LLM integration
- Centralized prompt management
- Basic RAG functionality
- Logging and error handling

Future updates will include:
- Enhanced document processing
- Additional model support
- Advanced retrieval strategies
- Performance optimizations