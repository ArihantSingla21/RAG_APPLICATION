import sys
import os 
import pandas as pd 
import numpy as np
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from src.components.prompts import get_rag_query_prompt

def load_documents(directory: str) -> List[Any]:
    """
    Load documents from a directory.
    
    Args:
        directory: Directory to load documents from
            
    Returns:
        List of loaded documents
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    logging.info(f"Loading documents from {directory}")
    documents = SimpleDirectoryReader(directory).load_data()
    logging.info(f"Loaded {len(documents)} documents")
    
    return documents

def get_embedding_model(embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Get embedding model based on the provided model name.
    
    Args:
        embedding_model_name: Name of the embedding model to use
        
    Returns:
        An embedding model instance
    """
    try:
        # Try using LangchainEmbedding first (more compatibility options)
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=embedding_model_name)
        )
        logging.info(f"Using LangchainEmbedding with {embedding_model_name}")
    except:
        # Fall back to HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        logging.info(f"Using HuggingFaceEmbedding with {embedding_model_name}")
    
    return embed_model

def build_index(
    documents: List[Any], 
    service_context: ServiceContext,
    persist_dir: Optional[str] = None
) -> VectorStoreIndex:
    """
    Build a vector store index from documents.
    
    Args:
        documents: List of documents
        service_context: ServiceContext instance with LLM and embedding model
        persist_dir: Directory to persist the index, if provided
            
    Returns:
        The built VectorStoreIndex
    """
    logging.info("Building index from documents")
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    
    # Save index if persist_dir is provided
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        logging.info(f"Index saved to {persist_dir}")
    
    return index

def load_index(
    persist_dir: str, 
    service_context: ServiceContext
) -> VectorStoreIndex:
    """
    Load a pre-built index from disk.
    
    Args:
        persist_dir: Directory where the index is persisted
        service_context: ServiceContext with LLM and embedding model
        
    Returns:
        The loaded VectorStoreIndex
    """
    if not os.path.exists(persist_dir):
        raise ValueError(f"Index directory {persist_dir} does not exist")
    
    logging.info(f"Loading index from {persist_dir}")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context
    )
    
    return index

def get_or_create_index(
    service_context: ServiceContext,
    persist_dir: str,
    data_dir: Optional[str] = None,
    documents: Optional[List[Any]] = None,
) -> VectorStoreIndex:
    """
    Get an existing index or create a new one if none exists.
    
    Args:
        service_context: ServiceContext with LLM and embedding model
        persist_dir: Directory where the index is or will be persisted
        data_dir: Directory containing documents (used if documents not provided)
        documents: List of documents (if already loaded)
        
    Returns:
        VectorStoreIndex for querying
    """
    # Try to load an existing index
    try:
        return load_index(persist_dir, service_context)
    except ValueError:
        # If loading fails, build a new index
        logging.info("No existing index found, building new index")
        if documents is None and data_dir is not None:
            documents = load_documents(data_dir)
        elif documents is None:
            raise ValueError("Either documents or data_dir must be provided")
        
        return build_index(documents, service_context, persist_dir)

def extract_source_documents(response) -> List[Dict[str, Any]]:
    """
    Extract source documents from a response.
    
    Args:
        response: Response from query engine
        
    Returns:
        List of source document details
    """
    source_documents = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            source_documents.append({
                'text': node.node.text,
                'score': node.score if hasattr(node, 'score') else None,
                'doc_id': node.node.doc_id if hasattr(node.node, 'doc_id') else None
            })
    
    return source_documents

def query_index(
    index: VectorStoreIndex,
    query_text: str,
    service_context: Optional[ServiceContext] = None,
    similarity_top_k: int = 3
) -> Dict[str, Any]:
    """
    Query the RAG index.
    
    Args:
        index: VectorStoreIndex to query
        query_text: The query text
        service_context: Optional ServiceContext (uses index's if not provided)
        similarity_top_k: Number of similar documents to retrieve
        
    Returns:
        Dict containing the response and additional information
    """
    # Get the query engine with appropriate settings
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        service_context=service_context,
        text_qa_template=get_rag_query_prompt()
    )
    
    # Execute the query
    logging.info(f"Executing query: {query_text}")
    response = query_engine.query(query_text)
    
    # Extract source documents
    source_documents = extract_source_documents(response)
    
    # Construct the result
    result = {
        'response': str(response),
        'source_documents': source_documents,
        'raw_response': response
    }
    
    return result

def create_service_context(llm, embed_model = None):
    """
    Create a service context with LLM and optional embedding model.
    
    Args:
        llm: LLM instance
        embed_model: Optional embedding model (will create one if not provided)
        
    Returns:
        ServiceContext instance
    """
    if embed_model is None:
        embed_model = get_embedding_model()
        
    return ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
