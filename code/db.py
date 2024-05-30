import asyncpg
from asyncpg import Connection

from code.config import settings

from loguru import logger


class AsyncpgConnection:
    _instance: Connection = None

    @classmethod
    async def connect(cls):
        if cls._instance is None:
            logger.info(f'Connecting to database: {settings.database_url}')
            cls._instance = await asyncpg.connect(dsn=str(settings.database_url))
        return cls._instance

    @classmethod
    async def disconnect(cls):
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

    @classmethod
    async def execute(cls, query: str, *args):
        return await cls._instance.execute(query, *args)

    @classmethod
    async def executemany(cls, query: str, *args):
        return await cls._instance.executemany(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args):
        return await cls._instance.fetch(query, *args)
