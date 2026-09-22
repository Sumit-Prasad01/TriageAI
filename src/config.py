import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemeni_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimension: str = os.getenv("EMBEDDING_DIMENSION", "3072")

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD")
    pinecone_region: str = os.getenv("PINECONE_REGION")

    # Internet search
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Self-RAG controls
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_support_retries: int = int(os.getenv("MAX_SUPPORT_RETRIES", "2"))
    max_retrieval_rewrites: int = int(os.getenv("MAX_RETRIEVAL_REWRITES", "2"))
    max_web_rewrites: int = int(os.getenv("MAX_WEB_REWRITES", "2"))
    database_path: str = os.getenv("DATABASE_PATH", "data/audit.db")


    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()