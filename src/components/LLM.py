"""
LLM Client Module for Hugging Face Inference API Integration
"""

import os
from typing import Optional, List, Dict, Any, AsyncGenerator, Iterator
from dotenv import load_dotenv
import requests
from llama_index.core.llms import LLM, ChatMessage, CompletionResponse, LLMMetadata
from pydantic import Field, PrivateAttr
import json
import asyncio
from src.exception import CustomException
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from functools import lru_cache
# from langchain.vectorstores import LangchainEmbedding

# Load environment variables
load_dotenv()

class HuggingFaceInferenceAPI(LLM):
    """LLM implementation for Hugging Face Inference API."""
    
    model_name: str = Field(description="Name of the model to use")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    max_new_tokens: int = Field(default=512, description="Maximum number of tokens to generate")
    
    _api_key: str = PrivateAttr()
    _api_url: str = PrivateAttr()
    _headers: dict = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    _query_template: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        system_prompt: str = "You are a helpful AI assistant.",
        query_template: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        # Load API key from environment
        load_dotenv()
        self._api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self._api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
        # Set up API configuration
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._headers = {"Authorization": f"Bearer {self._api_key}"}
        
        # Store prompts
        self._system_prompt = system_prompt
        self._query_template = query_template or "{query_str}"

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt."""
        try:
            # Format the prompt with system prompt and query template
            if "{query_str}" in self._query_template:
                formatted_prompt = f"{self._system_prompt}\n\n{self._query_template.format(query_str=prompt)}"
            else:
                formatted_prompt = f"{self._system_prompt}\n\n{prompt}"

            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "return_full_text": False
                }
            }

            response = requests.post(self._api_url, headers=self._headers, json=payload)
            response.raise_for_status()
            
            text = response.json()[0].get("generated_text", "")
            
            return CompletionResponse(text=text)
            
        except Exception as e:
            raise Exception(f"Error in API call: {str(e)}")

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """
        Generate a response to a list of chat messages.
        
        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional parameters for the API call
            
        Returns:
            ChatMessage object
        """
        try:
            # Convert messages to a format suitable for the model
            conversation = []
            
            # Process each message
            for msg in messages:
                if hasattr(msg, 'content'):  # Direct ChatMessage object
                    content = msg.content
                    role = msg.role
                else:  # Dict-like object
                    content = msg.get('content', '')
                    role = msg.get('role', 'user')
                
                prefix = "System: " if role == "system" else "Human: " if role == "user" else "Assistant: "
                conversation.append(f"{prefix}{content}")
            
            # Join the conversation with newlines
            full_prompt = "\n".join(conversation)
            full_prompt += "\nAssistant: "
            
            # Get completion
            completion_response = self.complete(full_prompt, **kwargs)
            
            # Return formatted response
            return ChatMessage(role="assistant", content=completion_response.text)
        
        except Exception as e:
            raise CustomException(f"Error in chat method: {str(e)}")

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Async version of complete."""
        return self.complete(prompt, **kwargs)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """Async version of chat."""
        return self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[CompletionResponse]:
        """Stream completion response."""
        response = self.complete(prompt, **kwargs)
        yield response

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Iterator[ChatMessage]:
        """Stream chat response."""
        response = self.chat(messages, **kwargs)
        yield response

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        """Async stream completion response."""
        response = await self.acomplete(prompt, **kwargs)
        yield response

    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[ChatMessage, None]:
        """Async stream chat response."""
        response = await self.achat(messages, **kwargs)
        yield response

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=2048,  # Typical context window size
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=False,
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_cached_embed_model(model_name: str):
        """Cache the embedding model to avoid reloading."""
        try:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            return LangchainEmbedding(embeddings)
        except Exception as e:
            raise CustomException(f"Error initializing embedding model: {str(e)}")

# Pre-configured models
AVAILABLE_MODELS = {
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "system_prompt": "You are a helpful AI assistant that provides accurate and detailed information."
    },
    "zephyr": {
        "model_name": "HuggingFaceH4/zephyr-7b-beta",
        "system_prompt": "You are a helpful AI assistant that provides accurate and detailed information."
    },
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "system_prompt": "You are a helpful AI assistant that provides accurate and detailed information."
    },
    "llama2-7b": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "system_prompt": "You are a helpful AI assistant that provides accurate and detailed information."
    }
}

def get_llm_client(
    model_key: str = "mistral-7b",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    system_prompt: str = "You are a helpful AI assistant.",
    query_template: Optional[str] = None,
) -> HuggingFaceInferenceAPI:
    """Get a configured LLM client."""
    
    # Map model keys to their Hugging Face model IDs
    model_map = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
        "zephyr": "HuggingFaceH4/zephyr-7b-beta",
        "phi-2": "microsoft/phi-2",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf"
    }
    
    model_name = model_map.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model key: {model_key}")
    
    return HuggingFaceInferenceAPI(
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        query_template=query_template
    )
