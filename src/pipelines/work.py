import os
from pathlib import Path
from typing import Optional, Dict, Any
from src.components.model import RAGModel
from src.components.LLM import get_llm_client
from src.exception import CustomException
import sys
from src.components.Data_ingestion import DataIngestion

class RAGPipeline:
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.pdf_dir = self.artifacts_dir / "data"
        self.index_dir = self.artifacts_dir / "index"
        self._setup_directories()
        self.model = None
        self.data_ingestion = DataIngestion()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def initialize_model(self, model_key: str = "mistral-7b") -> None:
        """Initialize the RAG model with specified parameters."""
        ## entry point inot model.py
        try:
            self.model = RAGModel(
                model_key=model_key,
                persist_dir=str(self.index_dir),
                data_dir=str(self.pdf_dir)
            )
        except Exception as e:
            raise CustomException(
                error_message=f"Error initializing RAG model: {str(e)}",
                error_detail=sys.exc_info()
            )

    def process_pdf(self, pdf_content: bytes, filename: str) -> bool:
        """Process a PDF file using DataIngestion component."""
        try:
            # Save PDF to artifacts directory
            pdf_path = self.pdf_dir / filename
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)

            # Use DataIngestion to load documents
            _, documents = self.data_ingestion.initiate_data_ingestion()

            # Initialize model if not already initialized
            if self.model is None:
                self.initialize_model()

            # Build index from the loaded documents
            self.model.build_index(documents=documents, save_index=True)
            return True

        except Exception as e:
            raise CustomException(
                error_message=f"Error processing PDF: {str(e)}",
                error_detail=sys.exc_info()
            )

    def query_pdf(self, question: str) -> Dict[str, Any]:
        """
        Query the processed PDF.
        
        Args:
            question: User's question about the PDF
            
        Returns:
            Dict containing response and source information
        """
        if self.model is None:
            raise CustomException(
                error_message="Model not initialized. Please process a PDF first.",
                error_detail=None
            )

        try:
            return self.model.query(question)
        except Exception as e:
            raise CustomException(
                error_message=f"Error querying model: {str(e)}",
                error_detail=sys.exc_info()
            )

    def clear_session(self):
        """Clear current session data."""
        self.model = None
