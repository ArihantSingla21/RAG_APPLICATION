import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from src.components.LLM import get_llm_client
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import shutil
from llama_index.core import SimpleDirectoryReader

@dataclass
class DataIngestionConfig:
    # Directory where processed PDFs will be stored
    data_dir: str = os.path.join("artifacts", "data")
    # Directory where the raw PDF files are located 
    raw_data_path: str = os.path.join("notebook", "DATA")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, List[Any]]:
        """Load documents from the data directory."""
        logging.info("Initiating data ingestion")
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs(self.ingestion_config.data_dir, exist_ok=True)
            
            # Load documents using SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                self.ingestion_config.data_dir
            ).load_data()
            
            logging.info(f"Loaded {len(documents)} documents")
            return self.ingestion_config.data_dir, documents
            
        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)

    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save an uploaded file to the data directory."""
        try:
            file_path = os.path.join(self.ingestion_config.data_dir, filename)
            os.makedirs(self.ingestion_config.data_dir, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logging.info(f"Saved uploaded file: {filename}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving uploaded file: {filename}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test the PDF ingestion
    obj = DataIngestion()
    data_dir, documents = obj.initiate_data_ingestion()
    print(f"Data directory: {data_dir}")
    print(f"Number of documents loaded: {len(documents)}")
    
    # Print first few characters of each document to verify
    for i, doc in enumerate(documents[:3]):  # Show first 3 docs only
        print(f"Document {i+1} preview: {doc.text[:100]}...")