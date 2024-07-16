import traceback

from typing import Any
from pydantic import PostgresDsn

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

import core.config as cfg
from core.logger import logger


# Подключение к базе данных

db_URI = PostgresDsn.build(
    scheme='postgresql+asyncpg',
    username=cfg.db_login,
    password=cfg.db_password,
    host=cfg.db_host,
    port=cfg.db_port,
    path=cfg.db_name
    # path=f"/{cfg.db_name}"
)

db_engine = create_async_engine(
    db_URI.unicode_string(),
    pool_pre_ping=True,
    echo=False,
    connect_args={
        'server_settings': {
            'application_name': cfg.app_name,
        }
    }
)

SessionLocal = sessionmaker(
    db_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
)


Base = declarative_base()


async def get_db() -> Any:
    async with SessionLocal() as session:
        yield session


# async def check_connection(
#     db_name: str,
#     login: str,
#     password: str,
#     host: str,
#     port: int
# ) -> bool:
#     """
#     Проверка подключения к базе данных
#     """
#     db_URI = PostgresDsn.build(
#         scheme='postgresql+asyncpg',
#         username=login,
#         password=password,
#         host=host,
#         port=str(port),
#         path=f"/{db_name}"
#     )
#     engine = create_async_engine(
#         db_URI,
#         pool_pre_ping=True,
#         pool_timeout=5,
#         connect_args={
#             'server_settings': {
#                 'application_name': cfg.app_name,
#                 # "options": f" -c statement_timeout={int(config.db_connection_timeout)*1000}ms",
#             }
#         }
#     )
#     SessionLocal = sessionmaker(
#         engine,
#         class_=AsyncSession,
#         autocommit=False,
#         autoflush=False
#     )
#     try:
#         async with SessionLocal() as session:
#             sql_string = text("SELECT 1;")
#             await session.execute(sql_string)
#             return True
#     except Exception as e:
#         logger.error(e)
#         traceback.print_exc()
#         return False
