# RAG Application

## Overview
This RAG (Retrieval-Augmented Generation) application leverages advanced natural language processing techniques to enhance the generation of text by retrieving relevant information from a knowledge base. Built with LlamaIndex and LangChain, it provides a robust foundation for creating intelligent document retrieval and question-answering systems.

## Project Structure
```
RAG_Application/
├── src/
│   ├── logger.py      # Logging configuration and utilities
│   └── utils.py       # Utility functions and helpers
├── logs/              # Directory for application logs
├── requirements.txt   # Project dependencies
└── setup.py          # Package setup configuration
```

## Features
- **Advanced Retrieval System**: Utilizes LlamaIndex for efficient document indexing and retrieval
- **Flexible Architecture**: Built on LangChain for extensible language model interactions
- **Vector Storage**: Integration with ChromaDB for efficient vector storage and similarity search
- **Robust Logging**: Comprehensive logging system with timestamped logs
- **Data Processing**: Integrated with Pandas and NumPy for efficient data handling
- **Scalable Design**: Built to handle large document collections efficiently
- **Customizable Embeddings**: Support for various embedding models and configurations

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

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Key Dependencies
- **LlamaIndex (v0.9.x)**: Core framework for building and querying knowledge bases
  - Provides document processing and indexing capabilities
  - Supports various index structures for efficient retrieval
- **LangChain**: Framework for developing applications with LLMs
  - Enables creation of complex language model chains
  - Provides tools for prompt management and output parsing
- **ChromaDB**: Vector database for storing and retrieving embeddings
  - Efficient similarity search capabilities
  - Persistent storage for embeddings
- **Pandas & NumPy**: Data processing libraries
  - Used for structured data handling
  - Provides numerical computing capabilities
- Additional dependencies are listed in `requirements.txt`

## Configuration
1. **Environment Variables**:
   Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_api_key_here
   CHROMA_DB_DIR=path/to/chromadb
   LOG_LEVEL=INFO
   ```

2. **Logging Configuration**:
   - Logs are automatically created in the `logs` directory
   - Each log file is named with timestamp format: `MM_DD_YYYY_HH_MM_SS.log`
   - Log level can be configured in the environment variables

## Usage

### 1. Basic Setup
```python
from src.utils import setup_environment
from src.logger import logging

# Initialize the environment
setup_environment()
logging.info("Environment initialized successfully")
```

### 2. Document Processing
```python
# Example code for processing documents will go here
# This section will be updated as the project develops
```

### 3. Running Queries
```python
# Example code for running queries will go here
# This section will be updated as the project develops
```

## Logging System
The application includes a comprehensive logging system:

1. **Log File Structure**:
   - Timestamp-based naming: `MM_DD_YYYY_HH_MM_SS.log`
   - Located in the `logs` directory
   - Automatic directory creation if not exists

2. **Log Format**:
   ```
   [timestamp] line_number name - level - message
   ```

3. **Log Levels**:
   - INFO: General operational messages
   - ERROR: Error conditions
   - WARNING: Warning messages
   - DEBUG: Detailed debugging information

## Error Handling
- Comprehensive error handling for API failures
- Graceful degradation when services are unavailable
- Detailed error logging for debugging

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
**Note**: This project is under active development. Features and documentation will be updated regularly.