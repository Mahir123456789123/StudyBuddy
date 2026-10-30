"""
StudyBuddy Configuration Module.
Loads settings from environment variables and .env file using Pydantic.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # LLM Backend selection: mock | hf_local | ollama | cloud
    LLM_BACKEND: Literal["mock", "hf_local", "ollama", "cloud"] = "mock"

    # Hugging Face Local Backend settings
    HF_MODEL_NAME_OR_PATH: str = "Qwen/Qwen2.5-0.5B-Instruct"
    LORA_ADAPTER_PATH: str = "models/adapter/"

    # Ollama Backend settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"

    # Cloud Backend settings (OpenAI-compatible)
    CLOUD_API_KEY: Optional[str] = None
    CLOUD_API_BASE: str = "https://api.openai.com/v1"
    CLOUD_MODEL_NAME: str = "gpt-4o-mini"

    # Ingestion & Retrieval Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH: str = "indexes/faiss_index"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    RETRIEVER_K: int = 3

    # Server & Validation Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    MAX_QUESTION_LENGTH: int = 1000
    MAX_STORED_QUIZZES: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global singleton instance
settings = Settings()
