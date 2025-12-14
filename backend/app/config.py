"""
FastAPI application configuration using Pydantic for environment variable validation.
This module centralizes all configuration settings and provides type safety.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # Production RAG Features
    use_hybrid_retrieval: bool = Field(default=True, alias="USE_HYBRID_RETRIEVAL")
    use_strict_validation: bool = Field(default=True, alias="USE_STRICT_VALIDATION")
    use_conversation_history: bool = Field(default=True, alias="USE_CONVERSATION_HISTORY")
    streaming_chunk_size_mb: int = Field(default=10, alias="STREAMING_CHUNK_SIZE_MB")
    rrf_k: int = Field(default=60, alias="RRF_K")
    mmr_lambda: float = Field(default=0.7, alias="MMR_LAMBDA")
    citation_threshold: float = Field(default=0.7, alias="CITATION_THRESHOLD")

    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8001, alias="APP_PORT")
    
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
    # CRITICAL: Change these in production! Generate with: openssl rand -hex 32
    jwt_secret_key: str = Field(
        default="test-secret-key-for-development-min-32-chars",
        alias="JWT_SECRET_KEY",
        description="JWT secret key (MUST be changed in production)"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    # CRITICAL: Change admin password in production! Minimum 16 characters, mixed case, numbers, symbols
    admin_password: str = Field(
        default="admin123",
        alias="ADMIN_PASSWORD",
        description="Admin password (MUST be changed in production, min 16 chars)"
    )
    
    def validate_security_settings(self):
        """Validate security settings and warn about insecure defaults."""
        warnings = []
        
        # Check JWT secret
        if self.jwt_secret_key == "test-secret-key-for-development-min-32-chars":
            warnings.append("⚠️  WARNING: Using default JWT_SECRET_KEY! Generate new key: openssl rand -hex 32")
        elif len(self.jwt_secret_key) < 32:
            warnings.append(f"⚠️  WARNING: JWT_SECRET_KEY too short ({len(self.jwt_secret_key)} chars). Use 32+ characters.")
        
        # Check admin password
        if self.admin_password == "admin123":
            warnings.append("⚠️  WARNING: Using default ADMIN_PASSWORD! Set a strong password (16+ chars).")
        elif len(self.admin_password) < 16:
            warnings.append(f"⚠️  WARNING: ADMIN_PASSWORD too short ({len(self.admin_password)} chars). Use 16+ characters.")
        
        # Check production mode
        if not self.debug_mode and warnings:
            # In production, these are CRITICAL
            raise ValueError(
                "CRITICAL SECURITY ERROR:\n" + "\n".join(warnings) +
                "\n\nProduction mode requires secure credentials. Set APP_ENV=development to bypass this check."
            )
        elif warnings:
            # In development, just warn
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("\n" + "="*80 + "\n" + "\n".join(warnings) + "\n" + "="*80)
    
    # LLM Configuration  
    lmstudio_api_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_API_URL")
    lmstudio_api_key: Optional[str] = Field(default="", alias="LMSTUDIO_API_KEY")
    lmstudio_model_name: str = Field(default="mistral-7b-instruct-v0.3", alias="LMSTUDIO_MODEL_NAME")
    llm_timeout_seconds: int = Field(default=300, alias="LLM_TIMEOUT_SECONDS")
    
    # RAG Configuration (Industrial-Grade Optimized)
    embedding_model_name: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL_NAME")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")  # bge-base: 768 dims
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2", alias="RERANKER_MODEL_NAME")  # Upgraded to 12-layer
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")  # Industrial: 800 tokens for better context
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")  # 25% overlap
    top_k_retrieval: int = Field(default=50, alias="TOP_K_RETRIEVAL")  # Cast wider net for precision
    top_k_rerank: int = Field(default=15, alias="TOP_K_RERANK")  # After reranking
    top_k_final: int = Field(default=3, alias="TOP_K_FINAL")  # More context for complex queries
    
    # Feature Flags
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    use_graph_search: bool = Field(default=True, alias="USE_GRAPH_SEARCH")
    use_ragas_evaluation: bool = Field(default=False, alias="USE_RAGAS_EVALUATION")  # Optional RAGAS metrics (Rec #8)
    
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
    
    # File Upload Settings (Industrial Grade)
    max_file_size_mb: int = Field(default=200, alias="MAX_FILE_SIZE_MB")  # Industrial: 200MB for large documents
    allowed_file_types: str = Field(default="pdf,docx,txt,html,json,csv,xlsx", alias="ALLOWED_FILE_TYPES")
    max_concurrent_uploads: int = Field(default=10, alias="MAX_CONCURRENT_UPLOADS")  # Parallel processing
    
    # Retry Configuration
    max_retry_attempts: int = Field(default=3, alias="MAX_RETRY_ATTEMPTS")
    retry_delay_seconds: int = Field(default=5, alias="RETRY_DELAY_SECONDS")
    retry_backoff_multiplier: float = Field(default=2.0, alias="RETRY_BACKOFF_MULTIPLIER")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=10, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    
    # Monitoring & Health
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")
    enable_performance_logging: bool = Field(default=True, alias="ENABLE_PERFORMANCE_LOGGING")
    
    # External APIs
    next_public_api_base: str = Field(default="http://localhost:8000", alias="NEXT_PUBLIC_API_BASE")
    next_public_ws_base: str = Field(default="ws://localhost:8000", alias="NEXT_PUBLIC_WS_BASE")
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    next_public_ai_sdk_key: Optional[str] = Field(default="", alias="NEXT_PUBLIC_AI_SDK_KEY")
    ai_sdk_key: Optional[str] = Field(default="", alias="AI_SDK_KEY")
    freeapi_key: Optional[str] = Field(default="", alias="FREEAPI_KEY")
    
    # CORS Configuration (for production security)
    cors_allowed_origins: Optional[str] = Field(default="", alias="CORS_ALLOWED_ORIGINS")
    
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


# Create settings instance and validate security
settings = Settings()

# Validate security settings on import (fail fast in production with weak credentials)
if hasattr(settings, 'validate_security_settings'):
    settings.validate_security_settings()
    @property
    def default_top_p(self) -> float:
        """Default top_p for LLM generation."""
        return 0.9


# Global settings instance
settings = Settings()