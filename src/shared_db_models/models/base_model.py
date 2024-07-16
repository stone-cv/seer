from typing import Any

from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logger import logger
from shared_db_models.database import Base


class BaseCRUD(Base):
    __abstract__ = True

    @classmethod
    async def get_by_id(
        cls,
        db_session: AsyncSession,
        id: int
    ) -> Any:

        result = await db_session.execute(
            select(cls).filter(
                cls.id == id
            )
        )

        result = result.scalars().first()
        return result

    @classmethod
    async def create(
        cls,
        db_session: AsyncSession,
        **kwargs
    ) -> Any:

        result = cls(**kwargs)

        db_session.add(result)
        await db_session.commit()

        logger.debug(f'DB record for {cls.__name__} created: {result.__dict__}')

        return result
    
    @classmethod
    async def update(
        cls,
        db_session: AsyncSession,
        id: int,
        **kwargs
    ) -> Any:

        result = await db_session.execute(
            select(cls).filter(
                cls.id == id
            )
        )
        result = result.scalars().first()

        for attr, value in kwargs.items():
            if hasattr(result, attr):
                setattr(result, attr, value)

        await db_session.commit()

        logger.debug(f'DB record for {cls.__name__} updated: {result.__dict__}')

        return result
    
    @classmethod
    async def delete(
        cls,
        db_session: AsyncSession,
        id: int
    ) -> Any:

        result = await db_session.execute(
            select(cls).filter(
                cls.id == id
            )
        )

        result = result.scalars().first()
        result.deleted = True
        await db_session.commit()

        logger.debug(f'DB record for {cls.__name__} ID {id} deleted')

        return result
    
    @classmethod
    async def check_if_exists(
        cls,
        db_session: AsyncSession,
        param_name: str,
        param_val: Any
    ) -> Any:
        
        if hasattr(cls, param_name):
            cls_param = getattr(cls, param_name)

            result = await db_session.execute(
                select(cls).filter(
                    cls_param == param_val
                )
            )
            result = result.scalars().first()
        else:
            raise AttributeError(f'Class {cls.__name__} has no attribute {param_name}')

        return result
