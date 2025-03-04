import os
import sys
from src.components.LLM import get_llm_client
from src.components.prompts import get_rag_query_prompt, get_system_prompt
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.logger import logging
from src.exception import CustomException

class RAGModel:
    """
    Retrieval-Augmented Generation model for document retrieval and response generation.
    """
    def __init__(
        self,
        model_key: str = "mistral-7b",
        system_prompt_type: str = "rag",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        persist_dir: str = "artifacts/index",
        data_dir: str = "artifacts/data",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        context_window: int = 2048,
    ):
        """
        Initialize the RAG model with Hugging Face Inference API.
        """
        try:
            # Get system prompt
            system_prompt = get_system_prompt(system_prompt_type)
            if not system_prompt:
                system_prompt = "You are a helpful AI assistant specialized in retrieving and analyzing information."
            
            # Get query template - fixed to call without arguments
            query_template = get_rag_query_prompt()
            
            # Initialize LLM client directly
            self.llm = get_llm_client(
                model_key=model_key,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                query_template=query_template
            )
            
            # Initialize embedding model
            self.embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name=embedding_model_name)
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.context_window = context_window
            
            # Set directories
            self.persist_dir = persist_dir
            self.data_dir = data_dir
            self.index = None
            
        except Exception as e:
            raise CustomException(
                error_message=f"Error initializing RAG model: {str(e)}",
                error_detail=sys.exc_info()
            )
    
    def load_documents(self, directory: Optional[str] = None) -> List[Any]:
        """Load documents from a directory."""
        try:
            dir_to_load = directory or self.data_dir
            if not os.path.exists(dir_to_load):
                raise ValueError(f"Directory {dir_to_load} does not exist")
            
            logging.info(f"Loading documents from {dir_to_load}")
            documents = SimpleDirectoryReader(dir_to_load).load_data()
            logging.info(f"Loaded {len(documents)} documents")
            
            return documents
        except Exception as e:
            raise CustomException(
                error_message=f"Error loading documents: {str(e)}",
                error_detail=sys.exc_info()
            )
    
    def build_index(self, documents: Optional[List[Any]] = None, save_index: bool = True) -> VectorStoreIndex:
        """Build a vector store index from documents."""
        try:
            if documents is None:
                documents = self.load_documents()
            
            logging.info("Building index from documents")
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=Settings
            )
            
            if save_index:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                logging.info(f"Index saved to {self.persist_dir}")
            
            return self.index
        except Exception as e:
            raise CustomException(
                error_message=f"Error building index: {str(e)}",
                error_detail=sys.exc_info()
            )
    
    def load_index(self) -> VectorStoreIndex:
        """Load a pre-built index from disk."""
        try:
            if not os.path.exists(self.persist_dir):
                raise ValueError(f"Index directory {self.persist_dir} does not exist")
            
            logging.info(f"Loading index from {self.persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(
                storage_context=storage_context,
                service_context=Settings
            )
            
            return self.index
        except Exception as e:
            raise CustomException(
                error_message=f"Error loading index: {str(e)}",
                error_detail=sys.exc_info()
            )
    
    def get_or_create_index(self) -> VectorStoreIndex:
        """Get an existing index or create a new one if none exists."""
        if self.index is not None:
            return self.index
        
        try:
            return self.load_index()
        except ValueError:
            logging.info("No existing index found, building new index")
            return self.build_index()
    
    def query(self, query_text: str, similarity_top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG model."""
        try:
            # Get or create the index
            index = self.get_or_create_index()
            
            # Get the query engine
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                service_context=Settings,
                text_qa_template=get_rag_query_prompt()
            )
            
            # Execute query
            logging.info(f"Executing query: {query_text}")
            response = query_engine.query(query_text)
            
            # Extract source documents
            source_documents = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_documents.append({
                        'text': node.node.text,
                        'score': node.score if hasattr(node, 'score') else None,
                        'doc_id': node.node.doc_id if hasattr(node.node, 'doc_id') else None
                    })
            
            return {
                'response': str(response),
                'source_documents': source_documents,
                'raw_response': response
            }
            
        except Exception as e:
            raise CustomException(
                error_message=f"Error querying model: {str(e)}",
                error_detail=sys.exc_info()
            )

def get_rag_model(
    model_key: str = "mistral-7b",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v1.5",
    **kwargs
) -> RAGModel:
    """Get a RAG model with specified parameters."""
    try:
        return RAGModel(
            model_key=model_key,
            embedding_model_name=embedding_model,
            **kwargs
        )
    except Exception as e:
        raise CustomException(
            error_message=f"Error creating RAG model: {str(e)}",
            error_detail=sys.exc_info()
        )
