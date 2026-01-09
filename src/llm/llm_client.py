"""
LLM Client

Handles communication with local LLM (via Ollama) for:
- Community summarization
- Answer generation
"""

import requests
import json
from typing import Optional, List, Dict


class OllamaLLM:
    """
    Client for interacting with Ollama local LLM.
    
    Supports models like:
    - llama3:8b
    - mistral:7b
    - gemma2:9b
    """
    
    def __init__(
        self,
        model_name: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Ollama API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"{base_url}/api/generate"
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""
    
    def is_available(self) -> bool:
        """
        Check if Ollama is running and model is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            return self.model_name in model_names
        
        except:
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        
        except:
            return []


class MockLLM:
    """
    Mock LLM for testing without Ollama.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock LLM."""
        pass
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            max_tokens: Ignored
            temperature: Ignored
            system_prompt: Ignored
            
        Returns:
            Mock generated text
        """
        # Simple mock: return a generic response
        if "summary" in prompt.lower() or "community" in prompt.lower():
            return "This community focuses on key entities and their relationships in Ambedkar's works."
        else:
            return "Based on the provided context, this is a mock response about Dr. B.R. Ambedkar's contributions."
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True
    
    def list_models(self) -> List[str]:
        """Return mock model list."""
        return ["mock-model"]


def create_llm_client(
    model_name: str = "llama3:8b",
    base_url: str = "http://localhost:11434",
    use_mock: bool = False,
    **kwargs
) -> OllamaLLM:
    """
    Factory function to create LLM client.
    
    Args:
        model_name: Name of the Ollama model
        base_url: Ollama API base URL
        use_mock: If True, return MockLLM
        **kwargs: Additional arguments
        
    Returns:
        LLM client instance
    """
    if use_mock:
        return MockLLM()
    
    client = OllamaLLM(model_name=model_name, base_url=base_url, **kwargs)
    
    # Check if Ollama is available
    if not client.is_available():
        print(f"Warning: Ollama model '{model_name}' not available.")
        print(f"Available models: {client.list_models()}")
        print("Falling back to MockLLM for demo purposes.")
        return MockLLM()
    
    return client


def demo():
    """Demo LLM client."""
    # Try to connect to Ollama
    llm = create_llm_client()
    
    print("\n" + "="*60)
    print("LLM CLIENT DEMO:")
    print("="*60)
    print(f"Using: {type(llm).__name__}")
    print(f"Available: {llm.is_available()}")
    
    # Generate text
    prompt = "Summarize Dr. B.R. Ambedkar's contributions in one sentence."
    print(f"\nPrompt: {prompt}")
    
    response = llm.generate(prompt, max_tokens=100)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    demo()
