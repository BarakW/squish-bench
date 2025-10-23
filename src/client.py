from pydantic import Field
from pydantic_settings import BaseSettings


class ClientConfig(BaseSettings):
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_key: str = Field(alias="OPENROUTER_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


client_config = ClientConfig()  # type: ignore
