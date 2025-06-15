import ollama
import json
import time
from typing import Dict, Any, Optional

class OllamaClient:
    """
    A client wrapper for Ollama API that provides OpenAI-compatible interface
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        """
        Initialize Ollama client

        Args:
            model_name: Name of the Ollama model to use (default: llama3.1:8b)
            host: Ollama server host (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        
        # Check if model is available, if not try to pull it
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Check if the model is available locally, if not try to pull it"""
        try:
            # List available models
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]

            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found locally.")
                print("Available models:", model_names if model_names else "None")
                print(f"Please download the model manually: ollama pull {self.model_name}")
                print("Or use a different model that's already available.")

        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
            print("Please ensure Ollama is running and a model is available")
    
    def chat_completion(self, messages: list, temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a chat completion using Ollama

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Dictionary with completion response
        """
        try:
            # Convert messages to Ollama format
            prompt = self._convert_messages_to_prompt(messages)

            # Set up options
            options = {
                'temperature': temperature,
            }
            if max_tokens:
                options['num_predict'] = max_tokens

            # Make the request
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=options
            )

            # Format response to match OpenAI structure
            formatted_response = {
                'choices': [{
                    'message': {
                        'content': response['response'].strip(),
                        'role': 'assistant'
                    },
                    'finish_reason': 'stop'
                }],
                'model': self.model_name,
                'usage': {
                    'prompt_tokens': response.get('prompt_eval_count', 0),
                    'completion_tokens': response.get('eval_count', 0),
                    'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                }
            }

            return formatted_response

        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to a single prompt string"""
        prompt_parts = []
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

# Create a default client instance
default_client = None

def get_ollama_client(model_name: str = "llama3.1:8b") -> OllamaClient:
    """Get or create an Ollama client instance"""
    global default_client
    if default_client is None or default_client.model_name != model_name:
        default_client = OllamaClient(model_name=model_name)
    return default_client

def get_ollama_response(messages: list, model_name: str = "llama3.1:8b", temperature: float = 0.7) -> str:
    """
    Convenience function to get a response from Ollama
    
    Args:
        messages: List of message dictionaries
        model_name: Ollama model to use
        temperature: Sampling temperature
        
    Returns:
        Response content as string
    """
    client = get_ollama_client(model_name)
    response = client.chat_completion(messages, temperature=temperature)
    return response['choices'][0]['message']['content']
