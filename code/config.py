from pydantic import PostgresDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = True
    database_url: PostgresDsn


settings = Settings()
