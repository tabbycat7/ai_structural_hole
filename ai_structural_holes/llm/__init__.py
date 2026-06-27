"""LLM access layer (OpenRouter, OpenAI-compatible) with caching and a mock."""

from .client import LLMResponse, OpenRouterClient, MockClient, get_client

__all__ = ["LLMResponse", "OpenRouterClient", "MockClient", "get_client"]
