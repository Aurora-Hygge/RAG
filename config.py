import os
from dataclasses import dataclass
import dotenv

#加载环境变量
dotenv.load_dotenv()
@dataclass
class Settings:
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "chroma.db")
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-mpnet-base-v2")
    QWEN_API_KEY:str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL:str = os.getenv("QWEN_BASE_URL", "")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "")
    BAIDU_API_URL: str = os.getenv("BAIDU_API_URL", "")
    BAIDU_API_KEY: str = os.getenv("BAIDU_API_KEY", "")
    TIMEOUT: int = os.getenv("TIMEOUT", "")
    BAIDU_HOST: str = os.getenv("BAIDU_HOST", "")
    TEMP_PATH: str = os.getenv("TEMP_PATH", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")
    MEMORY: int = os.getenv("MEMORY", "")

settings = Settings()



