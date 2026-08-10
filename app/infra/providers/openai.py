from langchain_openai import ChatOpenAI 
from app.config.settings import get_settings


settings = get_settings()


def create_openai_client() -> ChatOpenAI:
    """Create a new OpenAI client instance.

    Overrides the SDK's default User-Agent ("AsyncOpenAI/Python x.y.z") —
    some custom/proxy base_urls run a WAF that blocks that exact string
    (403 "Your request was blocked."), independent of auth or payload.
    """
    return ChatOpenAI(
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_base_url or None,
        api_key=settings.llm_api_key or None,
        default_headers={"User-Agent": "bot-assistant/1.0"},
    )