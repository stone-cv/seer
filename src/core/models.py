import yaml
from typing import Optional

from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
# from core.security import fernet_encrypt


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    type_id: Mapped[int] = mapped_column(ForeignKey("event_types.id"))
    event_type: Mapped['EventType'] = relationship(back_populates="event")  # ?

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="event")

    camera_roi_id: Mapped[int] = mapped_column(ForeignKey("camera_rois.id"))
    camera_roi: Mapped['RegionOfInterest'] = relationship(back_populates="event")

    start_time: Mapped[DateTime] = mapped_column(DateTime)  # timestamp?
    end_time: Mapped[DateTime] = mapped_column(DateTime)  # timestamp?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # @staticmethod
    # async def client_get_by_name(
    #     db_session: AsyncSession, name: str
    # ) -> "ClientDatabase":
    #     result = await db_session.execute(
    #         select(ClientDatabase).filter(ClientDatabase.name == name)
    #     )

    #     return result.scalars().first()


class EventType(Base):
    __tablename__ = "event_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="event_type")  # ?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="camera")

    # roi_id: Mapped[int] = mapped_column(ForeignKey("roi_zones.id"))
    # event_type: Mapped['EventType'] = relationship(back_populates="event")  # ?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class RegionOfInterest(Base):
    __tablename__ = "roi_zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    coords: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="camera_roi")

    # event: Mapped['Event'] = relationship(back_populates="event_type")  # ?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
