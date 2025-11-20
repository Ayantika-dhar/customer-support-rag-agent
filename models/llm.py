'''
import os
import sys
from langchain_groq import ChatGroq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # Initialize the Groq chat model with the API key
        groq_model = ChatGroq(
            api_key="",
            model="",
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
'''

import os
import sys
from langchain_groq import ChatGroq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import get_config


def get_chatgroq_model():
    """Initialize and return the Groq chat model, or None if no API key."""
    try:
        config = get_config()
        api_key = config.get("GROQ_API_KEY", "")
        model_name = config.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

        # If no API key, return None so UI can show a friendly message
        if not api_key:
            return None

        groq_model = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return groq_model

    except Exception as e:
        # Let the caller handle showing errors
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
