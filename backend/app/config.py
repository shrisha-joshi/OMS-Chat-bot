"""
FastAPI application configuration using Pydantic for environment variable validation.
This module centralizes all configuration settings and provides type safety.
"""

from pydantic import BaseSettings, Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_env: str = Field(default="development", env="APP_ENV")
    app_host: str = Field(default="127.0.0.1", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    
    # Database URLs
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db: str = Field(default="company_kb", env="MONGODB_DB")
    redis_url: str = Field(..., env="REDIS_URL")
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default="", env="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="kb", env="QDRANT_COLLECTION")
    
    # ArangoDB
    arangodb_url: str = Field(..., env="ARANGODB_URL")
    arangodb_user: str = Field(..., env="ARANGODB_USER")
    arangodb_password: str = Field(..., env="ARANGODB_PASSWORD")
    arangodb_db: str = Field(default="rag_graph", env="ARANGODB_DB")
    
    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM Configuration
    lmstudio_api_url: str = Field(..., env="LMSTUDIO_API_URL")
    lmstudio_api_key: Optional[str] = Field(default="", env="LMSTUDIO_API_KEY")
    
    # RAG Configuration
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL_NAME")
    chunk_size: int = Field(default=750, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=10, env="TOP_K_RETRIEVAL")
    
    # Feature Flags
    use_reranker: bool = Field(default=True, env="USE_RERANKER")
    use_graph_search: bool = Field(default=True, env="USE_GRAPH_SEARCH")
    
    # File Upload Settings
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: str = Field(default="pdf,docx,txt,html,json", env="ALLOWED_FILE_TYPES")
    
    # External APIs
    next_public_api_base: str = Field(default="http://localhost:8000", env="NEXT_PUBLIC_API_BASE")
    ai_sdk_key: Optional[str] = Field(default="", env="AI_SDK_KEY")
    freeapi_key: Optional[str] = Field(default="", env="FREEAPI_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def allowed_file_extensions(self) -> list[str]:
        """Get list of allowed file extensions."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def debug_mode(self) -> bool:
        """Check if application is in debug mode."""
        return self.app_env.lower() == "development"


# Global settings instance
settings = Settings()