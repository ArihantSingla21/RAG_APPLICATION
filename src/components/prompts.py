from typing import Dict, Optional, List
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.prompts.base import PromptTemplate
import os

class PromptTemplates:
    """
    A collection of prompt templates for various LLM tasks.
    """
    
    # System prompts for different purposes
    SYSTEM_PROMPTS = {
        "rag": "You are a helpful assistant that provides accurate answers based on the given documents. "
              "Respond based on the context provided. If you don't know the answer, say so instead of making up information.",
        
        "chat": "You are a friendly and knowledgeable assistant. Provide helpful, accurate, and concise answers to the user's questions.",
        
        "summarization": "You are an expert at summarizing documents. Provide clear, concise summaries that capture the key points while maintaining accuracy.",
        
        "code": "You are a coding assistant with expertise in multiple programming languages. Provide clear, efficient code solutions and explanations."
    }
    
    # Query prompt templates for different tasks
    QUERY_TEMPLATES = {
        "rag": """
Below is a question:
Question: {query_str}
Please provide an answer based on the context:
Answer: """}

    
    @staticmethod
    def get_system_prompt(prompt_type: str = "rag", custom_prompt: Optional[str] = None) -> str:
        """
        Get a system prompt by type or use a custom prompt.
        """
        if custom_prompt:
            return custom_prompt
        
        return PromptTemplates.SYSTEM_PROMPTS.get(prompt_type, PromptTemplates.SYSTEM_PROMPTS["rag"])

    @staticmethod
    def get_query_prompt(prompt_type: str = "rag", custom_template: Optional[str] = None) -> SimpleInputPrompt:
        """
        Get a query prompt template by type or use a custom template.
        """
        template = custom_template if custom_template else PromptTemplates.QUERY_TEMPLATES.get(prompt_type, PromptTemplates.QUERY_TEMPLATES["rag"])
        return SimpleInputPrompt(template)


# These are the specific functions needed to resolve the import error
def get_rag_query_prompt() -> SimpleInputPrompt:
    """Get the default RAG query prompt."""
    return PromptTemplates.get_query_prompt("rag")

def get_system_prompt(prompt_type: str = "rag", custom_prompt: Optional[str] = None) -> str:
    """Get a system prompt by type or use a custom prompt."""
    return PromptTemplates.get_system_prompt(prompt_type, custom_prompt)