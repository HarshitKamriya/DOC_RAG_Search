"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env (local dev)
load_dotenv()

def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key) or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")

class Config:
    """Configuration class for RAG system"""
    
    # Model Configuration
    LLM_MODEL = "groq/llama-3.1-8b-instant"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        groq_key = _get_secret("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = groq_key
        return init_chat_model(cls.LLM_MODEL)
