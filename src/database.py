import traceback

from typing import Any
from pydantic import PostgresDsn

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

import config as cfg
from logger import logger


# Подключение к базе данных бота
bot_engine = create_async_engine(
    cfg.bot_db_URI,
    pool_pre_ping=True,
    connect_args={
        'server_settings': {
            'application_name': cfg.app_name,
        }
    }
)

BotSessionLocal = sessionmaker(
    bot_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
)


Base = declarative_base()


async def check_connection(
    db_name: str,
    login: str,
    password: str,
    host: str,
    port: int
) -> bool:
    """
    Проверка подключения к базе данных
    """
    db_URI = PostgresDsn.build(
        scheme='postgresql+asyncpg',
        user=login,
        password=password,
        host=host,
        port=str(port),
        path=f"/{db_name}"
    )
    engine = create_async_engine(
        db_URI,
        pool_pre_ping=True,
        pool_timeout=5,
        connect_args={
            'server_settings': {
                'application_name': cfg.app_name,
                # "options": f" -c statement_timeout={int(config.db_connection_timeout)*1000}ms",
            }
        }
    )
    SessionLocal = sessionmaker(
        engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False
    )
    try:
        async with SessionLocal() as session:
            sql_string = text("SELECT 1;")
            await session.execute(sql_string)
            return True
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return False


async def get_db() -> Any:
    async with BotSessionLocal() as session:
        yield session
