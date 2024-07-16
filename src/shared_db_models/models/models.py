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

import src.core.config as cfg
from src.core.logger import logger
from shared_db_models.database import Base
from shared_db_models.models.base_model import BaseCRUD


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    track_id: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="camera")
    video_file: Mapped['VideoFile'] = relationship(back_populates="camera")
    daily_cam_check: Mapped['DailyCamCheck'] = relationship(back_populates="camera")

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
    
    @staticmethod
    async def get_track_id_by_camera_id(
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

        track_id = camera.track_id
        logger.debug(f'Camera {camera_id} track_id retrieved: {track_id} ({type(track_id)})')

        return track_id


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    type_id: Mapped[int] = mapped_column(ForeignKey("event_types.id"))
    event_type: Mapped['EventType'] = relationship(back_populates="event")

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="event")

    time: Mapped[DateTime] = mapped_column(DateTime)

    machine: Mapped[str] = mapped_column(String, nullable=True)
    stone_number: Mapped[int] = mapped_column(Integer, nullable=True)
    stone_area: Mapped[str] = mapped_column(String, nullable=True)
    comment: Mapped[str] = mapped_column(String, nullable=True)

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
        stone_area: Optional[str] = '0',
        comment: Optional[str] = 'см2'
    ) -> 'Event':

        event = Event(
            type_id=type_id,
            camera_id=camera_id,
            time=time,
            machine=machine,
            stone_number=stone_number,
            stone_area=stone_area,
            comment=comment
        )

        db_session.add(event)
        await db_session.commit()

        logger.debug(f'Event created: {event.__dict__}')

        return event
    
    @staticmethod
    async def event_update_stone_area(
        *,
        db_session: AsyncSession,
        event_id: int,
        stone_area: str
    ) -> str:

        event = await db_session.execute(
            select(Event).filter(
                Event.id == event_id
            )
        )
        event = event.scalars().first()
        event.stone_area = stone_area
        logger.info(f'Event {event_id} stone area updated: {event.stone_area}')

        await db_session.commit()

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

        if not event.stone_area:
            event.stone_area = 0

        event_dict = {
            "date": event.time.strftime('%Y-%m-%d %H:%M:%S'),
            "machine": "PW1TK 3000",
            "operation": event_type.name,
            "number": "0",
            "area": float(event.stone_area),
            "comment": "площадь в см2"
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

    event: Mapped['Event'] = relationship(back_populates="event_type")

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


class VideoFile(BaseCRUD):
    __tablename__ = "video_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="video_file")

    path: Mapped[str] = mapped_column(String, nullable=False)
    playback_uri: Mapped[str] = mapped_column(String, nullable=True)
    vid_start: Mapped[DateTime] = mapped_column(DateTime, nullable=False)  # timestamp?
    vid_end: Mapped[DateTime] = mapped_column(DateTime, nullable=False)  # timestamp?

    is_downloaded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    download_start: Mapped[DateTime] = mapped_column(DateTime, nullable=True)  # timestamp?
    download_end: Mapped[DateTime] = mapped_column(DateTime, nullable=True)  # timestamp?

    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    det_start: Mapped[DateTime] = mapped_column(DateTime, nullable=True)  # timestamp?
    det_end: Mapped[DateTime] = mapped_column(DateTime, nullable=True)  # timestamp?

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    @staticmethod
    async def get_unprocessed_files(
        *,
        db_session: AsyncSession
    ) -> str:

        files = await db_session.execute(
            select(VideoFile).filter(
                VideoFile.is_downloaded == True,
                VideoFile.is_processed == False
            ).order_by(VideoFile.vid_start)
        )
        files = files.scalars().all()

        logger.debug(f'{len(files)} file(s) retrieved for processing: {[file.__dict__ for file in files]}')

        return files
    

class DailyCamCheck(Base):
    __tablename__ = "daily_cam_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="daily_cam_check")

    date: Mapped[DateTime] = mapped_column(DateTime)
    is_processed: Mapped[bool] = mapped_column(Boolean)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def daily_cam_check_create(
        *,
        db_session: AsyncSession,
        camera_id: int,
        date: datetime.date,
        is_processed: bool,
        deleted: bool = False
    ) -> 'DailyCamCheck':

        data = DailyCamCheck(
            camera_id=camera_id,
            date=date,
            is_processed=is_processed,
            deleted=deleted
        )

        db_session.add(data)
        await db_session.commit()

        logger.info(f'Daily check complete for camera {camera_id}: {is_processed}')
        logger.debug(f'Daily one cam check event: {data.__dict__}')

        return data


class DailyAllCamCheck(Base):
    __tablename__ = "daily_all_cam_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    date: Mapped[DateTime] = mapped_column(DateTime)
    is_processed: Mapped[bool] = mapped_column(Boolean)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def daily_all_cam_check_create(
        *,
        db_session: AsyncSession,
        date: datetime.date,
        is_processed: bool,
        deleted: bool = False
    ) -> 'DailyAllCamCheck':

        data = DailyAllCamCheck(
            date=date,
            is_processed=is_processed,
            deleted=deleted
        )

        db_session.add(data)
        await db_session.commit()

        logger.info(f'Daily check complete for all cameras: {is_processed}')
        logger.debug(f'Daily all cam check event: {data.__dict__}')

        return data
