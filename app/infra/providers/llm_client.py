from app.config.settings import get_settings
from app.infra.providers.openai import create_openai_client
from langchain_core.language_models import BaseChatModel

def create_llm_client() -> BaseChatModel    :
    settings = get_settings()
    
    if settings.llm_provider == "openai":
        return create_openai_client()   
    elif settings.llm_provider == "anthropic":
        # from app.infra.providers.anthropic import create_anthropic_client
        # return create_anthropic_client()
        return create_openai_client() 
    elif settings.llm_provider == "ollama":
        # from app.infra.providers.ollama import create_ollama_client
        # return create_ollama_client()
        return create_openai_client() 
    elif settings.llm_provider == "gemini":
        # from app.infra.providers.gemini import create_gemini_client
        # return create_gemini_client()
        return create_openai_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    