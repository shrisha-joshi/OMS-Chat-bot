"""
FastAPI application configuration using Pydantic for environment variable validation.
This module centralizes all configuration settings and provides type safety.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    
    # Database URLs (optional for testing)
    mongodb_uri: Optional[str] = Field(default="", alias="MONGODB_URI")
    mongodb_db: str = Field(default="company_kb", alias="MONGODB_DB")
    redis_url: Optional[str] = Field(default="", alias="REDIS_URL")
    qdrant_url: Optional[str] = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="kb", alias="QDRANT_COLLECTION")
    
    # ArangoDB (optional for testing)
    arangodb_url: Optional[str] = Field(default="", alias="ARANGODB_URL")
    arangodb_user: Optional[str] = Field(default="", alias="ARANGODB_USER")
    arangodb_password: Optional[str] = Field(default="", alias="ARANGODB_PASSWORD")
    arangodb_db: str = Field(default="rag_graph", alias="ARANGODB_DB")
    
    # Neo4j Configuration
    neo4j_uri: Optional[str] = Field(default="", alias="NEO4J_URI")
    neo4j_username: Optional[str] = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: Optional[str] = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: Optional[str] = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_disabled: bool = Field(default=False, alias="NEO4J_DISABLED")
    
    # Security
    jwt_secret_key: str = Field(default="test-secret-key-for-development-min-32-chars", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM Configuration  
    lmstudio_api_url: str = Field(default="http://192.168.56.1:1234/v1", alias="LMSTUDIO_API_URL")
    lmstudio_api_key: Optional[str] = Field(default="", alias="LMSTUDIO_API_KEY")
    lmstudio_model_name: str = Field(default="mistral-7b-instruct-v0.3", alias="LMSTUDIO_MODEL_NAME")
    
    # RAG Configuration
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_NAME")
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    
    # Feature Flags
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    use_graph_search: bool = Field(default=True, alias="USE_GRAPH_SEARCH")
    
    # Context Management
    max_context_tokens: int = Field(default=1500, alias="MAX_CONTEXT_TOKENS")
    max_llm_output_tokens: int = Field(default=1024, alias="MAX_LLM_OUTPUT_TOKENS")
    
    # Document Training & RAG Quality
    force_document_usage: bool = Field(default=False, alias="FORCE_DOCUMENT_USAGE")  # Allow responses without docs
    min_similarity_threshold: float = Field(default=0.3, alias="MIN_SIMILARITY_THRESHOLD")
    validate_document_usage: bool = Field(default=True, alias="VALIDATE_DOCUMENT_USAGE")
    fallback_on_no_documents: bool = Field(default=True, alias="FALLBACK_ON_NO_DOCUMENTS")
    require_citations: bool = Field(default=False, alias="REQUIRE_CITATIONS")  # Allow responses without citations
    reject_generic_responses: bool = Field(default=False, alias="REJECT_GENERIC_RESPONSES")  # Allow generic responses
    min_citation_count: int = Field(default=0, alias="MIN_CITATION_COUNT")  # No minimum citations required
    
    # Media Extraction & Enrichment
    extract_images_from_pdf: bool = Field(default=True, alias="EXTRACT_IMAGES_FROM_PDF")
    extract_video_links: bool = Field(default=True, alias="EXTRACT_VIDEO_LINKS")
    suggest_related_media: bool = Field(default=True, alias="SUGGEST_RELATED_MEDIA")
    max_suggested_media: int = Field(default=5, alias="MAX_SUGGESTED_MEDIA")
    
    # File Upload Settings
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_file_types: str = Field(default="pdf,docx,txt,html,json", alias="ALLOWED_FILE_TYPES")
    
    # External APIs
    next_public_api_base: str = Field(default="http://localhost:8000", alias="NEXT_PUBLIC_API_BASE")
    next_public_ai_sdk_key: Optional[str] = Field(default="", alias="NEXT_PUBLIC_AI_SDK_KEY")
    ai_sdk_key: Optional[str] = Field(default="", alias="AI_SDK_KEY")
    freeapi_key: Optional[str] = Field(default="", alias="FREEAPI_KEY")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
    
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
    
    @property 
    def default_temperature(self) -> float:
        """Default temperature for LLM generation."""
        return 0.7
    
    @property
    def default_top_p(self) -> float:
        """Default top_p for LLM generation."""
        return 0.9


# Global settings instance
settings = Settings()