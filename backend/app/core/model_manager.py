"""
Model Manager Singleton
Centralizes loading of heavy ML models (SentenceTransformer, CrossEncoder, Spacy)
to prevent memory leaks and redundant reloading.
"""

import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
import tiktoken
from ..config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.embedding_model = None
        self.reranker_model = None
        self.nlp_model = None
        self.tokenizer = None
        self._initialized = True
    
    async def initialize(self):
        """Initialize all models if not already loaded."""
        if self.embedding_model:
            return

        logger.info("🧠 ModelManager: Loading models...")
        
        # Load Embedding Model
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model_name}")
            self.embedding_model = SentenceTransformer(settings.embedding_model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Load Reranker Model
        if settings.use_reranker:
            try:
                logger.info(f"Loading reranker model: {settings.reranker_model_name}")
                self.reranker_model = CrossEncoder(settings.reranker_model_name)
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")

        # Load NLP Model (Spacy)
        try:
            logger.info("Loading spaCy model: en_core_web_sm")
            self.nlp_model = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Entity extraction will be limited.")
            self.nlp_model = None

        # Load Tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

        logger.info("✅ ModelManager: All models loaded successfully")

    def get_embedding_model(self):
        return self.embedding_model

    def get_reranker_model(self):
        return self.reranker_model

    def get_nlp_model(self):
        return self.nlp_model

    def get_tokenizer(self):
        return self.tokenizer

# Global instance
model_manager = ModelManager()

async def get_model_manager():
    await model_manager.initialize()
    return model_manager
