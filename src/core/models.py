import yaml
import datetime
from typing import Optional

from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
# from core.security import fernet_encrypt


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    type_id: Mapped[int] = mapped_column(ForeignKey("event_types.id"))  # 'operation' as per api
    event_type: Mapped['EventType'] = relationship(back_populates="event")  # ?

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="event")

    # camera_roi_id: Mapped[int] = mapped_column(ForeignKey("camera_rois.id"))
    # camera_roi: Mapped['RegionOfInterest'] = relationship(back_populates="event")

    time: Mapped[DateTime] = mapped_column(DateTime)  # timestamp? 'date' as per api

    machine: Mapped[str] = mapped_column(String, nullable=True)  # station No. as per api
    stone_number: Mapped[int] = mapped_column(Integer, nullable=True)  # stone block No. as per api
    comment: Mapped[str] = mapped_column(String, nullable=True)  # other comment as per api

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def event_create(
        *,
        db_session: AsyncSession,
        type_id: int,
        camera_id: int,
        time: datetime
    ) -> 'Event':

        # event = await db_session.execute(
        #     select(Event)
        #     .filter(
        #         Event.name == name
        #     )
        # )
        # event = event.scalars().first()

        # if event is None:
        event = Event(
            type_id=type_id,
            camera_id=camera_id,
            time=time
        )
        # else:
        #     event.deleted = False

        db_session.add(event)
        await db_session.commit()

        return event

    @staticmethod
    async def event_delete(
        *,
        db_session: AsyncSession,
        event_id: int
    ) -> 'Event':

        event = await db_session.execute(
            select(Event).filter(
                Event.id == event_id
            )
        )

        event = event.scalars().first()
        event.deleted = True
        await db_session.commit()

        return event


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

    roi: Mapped[str] = mapped_column(String(255))

    # roi_id: Mapped[int] = mapped_column(ForeignKey("roi_zones.id"))
    # event_type: Mapped['EventType'] = relationship(back_populates="event")  # ?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


# class RegionOfInterest(Base):  # for multiple regions per camera
#     __tablename__ = "roi_zones"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     name: Mapped[str] = mapped_column(String(255))
#     coords: Mapped[str] = mapped_column(String(255))

#     event: Mapped['Event'] = relationship(back_populates="camera_roi")

#     # event: Mapped['Event'] = relationship(back_populates="event_type")  # ?

#     deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
