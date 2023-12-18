import yaml
from typing import Optional

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

import config as config
from database import Base
# from core.security import fernet_encrypt


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    type_id = Column(Integer, nullable=False)

    start_time = Column(DateTime)
    end_time = Column(DateTime)

    deleted = Column(Boolean, default=False, nullable=False)

    # @staticmethod
    # async def client_get_by_name(
    #     db_session: AsyncSession, name: str
    # ) -> "ClientDatabase":
    #     result = await db_session.execute(
    #         select(ClientDatabase).filter(ClientDatabase.name == name)
    #     )

    #     return result.scalars().first()