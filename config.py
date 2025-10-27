import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Settings:
    gemini_api_key: str | None = None
    hf_api_key: str | None = None


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        hf_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
