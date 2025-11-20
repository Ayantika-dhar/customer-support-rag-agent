'''
import os

def get_config():
    """
    Central place to read all config / API keys.
    For now we only use Groq. Later we’ll add OpenAI, Gemini, etc.
    """
    config = {
        # Groq
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL_NAME": os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
    }
    return config
'''

import os

def get_config():
    """
    Central place to read all config / API keys.
    """
    config = {
        # Groq (chat LLM)
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL_NAME": os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),

        # Embeddings (local)
        "EMBEDDING_MODEL_NAME": os.getenv(
            "EMBEDDING_MODEL_NAME",
            "all-MiniLM-L6-v2",
        ),

        # Web search (Tavily)
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
    }
    return config

