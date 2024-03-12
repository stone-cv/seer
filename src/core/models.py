import re
import ast
import json
from typing import Optional
from datetime import datetime

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

import core.config as cfg
from core.logger import logger
from core.database import Base
# from core.security import fernet_encrypt


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    type_id: Mapped[int] = mapped_column(ForeignKey("event_types.id"))  # 'operation' as per api
    event_type: Mapped['EventType'] = relationship(back_populates="event")  # ?

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="event")

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
        time: datetime,
        machine: Optional[str] = 'PW1TK 3000',
        stone_number: Optional[int] = 1,
        comment: Optional[str] = 'test'
    ) -> 'Event':

        event = Event(
            type_id=type_id,
            camera_id=camera_id,
            time=time,
            machine=machine,
            stone_number=stone_number,
            comment=comment
        )

        db_session.add(event)
        await db_session.commit()

        logger.debug(f'Event created: {event.__dict__}')

        return event
    
    @staticmethod
    async def convert_event_to_json(
        *,
        db_session: AsyncSession,
        event: 'Event'
    ):
        event_type = await EventType.get_type_by_id(
            db_session=db_session,
            type_id=event.type_id
        )

        event_dict = {
            "date": event.time.strftime('%Y-%m-%d %H:%M'),
            "machine": "PW1TK 3000",
            "operation": event_type.name,
            "number": "0",
            "comment": "Тестовый документ"
        }
        event_json = json.dumps(event_dict)
        return event_json

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

    @staticmethod
    async def get_type_by_id(
        *,
        db_session: AsyncSession,
        type_id: int
    ) -> 'EventType':

        event_type = await db_session.execute(
            select(EventType).filter(
                EventType.id == type_id
            )
        )

        event_type = event_type.scalars().first()
        return event_type


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="camera")

    roi: Mapped[str] = mapped_column(String(255))

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def camera_create(
        *,
        db_session: AsyncSession,
        name: str,
        url: str,
        roi_coord: str
    ) -> 'Camera':

        camera = Camera(
            name=name,
            url=url,
            roi_coord=roi_coord
        )

        db_session.add(camera)
        await db_session.commit()

        return camera

    @staticmethod
    async def camera_delete(
        *,
        db_session: AsyncSession,
        camera_id: int
    ) -> 'Camera':

        camera = await db_session.execute(
            select(Camera).filter(
                Camera.id == camera_id
            )
        )

        camera = camera.scalars().first()
        camera.deleted = True
        await db_session.commit()

        return camera
    
    @staticmethod
    async def update_camera_roi(
        *,
        db_session: AsyncSession,
        camera_id: int,
        roi_coord: str
    ) -> str:

        camera = await db_session.execute(
            select(Camera).filter(
                Camera.id == camera_id
            )
        )
        camera = camera.scalars().first()
        camera.roi = roi_coord
        logger.info(f'Camera {camera_id} ROI updated: {camera.roi}')

        await db_session.commit()

        return camera
    
    @staticmethod
    async def get_url_by_camera_id(
        *,
        db_session: AsyncSession,
        camera_id: int
    ) -> str:

        camera = await db_session.execute(
            select(Camera).filter(
                Camera.id == camera_id
            )
        )
        camera = camera.scalars().first()
        camera_url = camera.url
        logger.debug(f'Camera {camera_id} url retrieved: {camera_url}')

        pattern = r'rtsp://([^:]+):([^@]+)@'
        match = re.search(pattern, camera_url)

        try:
            if match:
                login = match.group(1)
                password = match.group(2)
                camera_url = camera_url.replace(f'{login}:{password}', f'{cfg.cam_login}:{cfg.cam_password}')
            else:
                logger.error(f'Invalid url for camera {camera_id}: {camera_url}\n{exc}')
    
        except Exception as exc:
            logger.error(exc)

        return camera_url
    
    @staticmethod
    async def get_roi_by_camera_id(
        *,
        db_session: AsyncSession,
        camera_id: int
    ) -> list:

        camera = await db_session.execute(
            select(Camera).filter(
                Camera.id == camera_id
            )
        )
        camera = camera.scalars().first()

        camera_roi = ast.literal_eval(camera.roi)
        logger.debug(f'Camera {camera_id} ROI retrieved: {camera_roi} ({type(camera_roi)})')

        return camera_roi


# class RegionOfInterest(Base):  # for multiple regions per camera
#     __tablename__ = "roi_zones"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     name: Mapped[str] = mapped_column(String(255))
#     coords: Mapped[str] = mapped_column(String(255))

#     event: Mapped['Event'] = relationship(back_populates="camera_roi")

#     # event: Mapped['Event'] = relationship(back_populates="event_type")  # ?

#     deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
