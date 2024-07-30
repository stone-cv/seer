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
from shared_db_models.database import Base
from shared_db_models.models.base_model import BaseCRUD


class Camera(BaseCRUD):
    """
    Модель камеры.
    Параметры:
        id: ID камеры в БД
        name: наименование камеры
        track_id: ID камеры в API видеорегистратора
        url: URL для скачивания видеофайлов
        roi: координаты области интереса
        deleted: флаг удаления камеры в БД
    """
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
    
    # @staticmethod
    # async def update_camera_roi(
    #     *,
    #     db_session: AsyncSession,
    #     camera_id: int,
    #     roi_coord: str
    # ) -> str:

    #     camera = await db_session.execute(
    #         select(Camera).filter(
    #             Camera.id == camera_id
    #         )
    #     )
    #     camera = camera.scalars().first()
    #     camera.roi = roi_coord
    #     logger.info(f'Camera {camera_id} ROI updated: {camera.roi}')

    #     await db_session.commit()

    #     return camera
    
    @staticmethod
    async def get_url_by_camera_id(
        *,
        db_session: AsyncSession,
        camera_id: int
    ) -> str:
        """
        Функция, позволяющая получить URL для скачивания видеофайлов
        по идентификатору камеры, а также добавить в URL авторизационные параметры.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            camera_id (int): ID камеры

        Returns:
            camera_url (str): URL для скачивания видеофайлов
        """
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
        """
        Функция, позволяющая получить координаты области интереса
        по идентификатору камеры.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            camera_id (int): ID камеры

        Returns:
            camera_roi (list): координаты области интереса
        """
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
        """
        Функция, позволяющая получить TrackID (ID камеры в API видеорегистратора)
        по идентификатору камеры.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            camera_id (int): ID камеры

        Returns:
            track_id (list): track_id
        """
        camera = await db_session.execute(
            select(Camera).filter(
                Camera.id == camera_id
            )
        )
        camera = camera.scalars().first()

        track_id = camera.track_id
        logger.debug(f'Camera {camera_id} track_id retrieved: {track_id} ({type(track_id)})')

        return track_id


class Event(BaseCRUD):
    """
    Модель события (срабатывания сценария по результатам детекции).
    Параметры:
        id: ID события
        type_id: ID типа события
        camera_id: ID камеры
        time: время срабатывания события
        machine: идентификатор станка на производстве
        stone_number: идентификатор анализируемого камня
        stone_area: площадь камня
        comment: комментарий
        deleted: флаг удаления в БД
    """
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    type_id: Mapped[int] = mapped_column(ForeignKey("event_types.id"))
    event_type: Mapped['EventType'] = relationship(back_populates="event")

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="event")

    time: Mapped[DateTime] = mapped_column(DateTime)

    machine: Mapped[str] = mapped_column(String, default='PW1TK 3000', nullable=True)
    stone_number: Mapped[int] = mapped_column(Integer, default=1, nullable=True)
    stone_area: Mapped[str] = mapped_column(String, default='0', nullable=True)
    comment: Mapped[str] = mapped_column(String, default='см2', nullable=True)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    @staticmethod
    async def convert_event_to_json(
        *,
        db_session: AsyncSession,
        event: 'Event'
    ):
        """
        Функция, позволяющая преобразовать объект события в формат JSON.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            event (Event): объект созданного события

        Returns:
            event_json (str): данные в формате JSON
        """
        event_type = await EventType.get_by_id(
            db_session=db_session,
            id=event.type_id
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


class EventType(BaseCRUD):
    """
    Модель типа события (срабатывания сценария по результатам детекции).
    Параметры:
        id: ID типа события
        name: наименование типа события
        deleted: флаг удаления в БД
    """
    __tablename__ = "event_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    event: Mapped['Event'] = relationship(back_populates="event_type")

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def get_id_by_name(
        db_session: AsyncSession,
        name: str
    ) -> int:
        """
        Функция, позволяющая получить ID типа события по его наименованию.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            name (str): наименование типа события

        Returns:
            event_type_id (int): ID типа события
        """
        event_type = await db_session.execute(
            select(EventType).filter(
                EventType.name == name
            )
        )
        event_type = event_type.scalars().first()
        logger.debug(f'Event type id for {name}: {event_type.id}')

        return event_type.id


class VideoFile(BaseCRUD):
    """
    Модель скачанного видеофайла.
    Параметры:
        id: ID видеофайла
        camera_id: ID камеры
        path: путь к видеофайлу
        playback_uri: URI для загрзки видеофайла
        vid_start: время начала видеофайла
        vid_end: время окончания видеофайла
        is_downloaded: флаг скачивания видеофайла
        download_start: время начала скачивания видеофайла
        download_end: время окончания скачивания видеофайла
        is_processed: флаг обработки видеофайла
        det_start: время начала обработки видеофайла
        det_end: время окончания обработки видеофайла
        deleted: флаг удаления в БД
    """
    __tablename__ = "video_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="video_file")

    path: Mapped[str] = mapped_column(String, nullable=False)
    playback_uri: Mapped[str] = mapped_column(String, nullable=True)
    vid_start: Mapped[DateTime] = mapped_column(DateTime, nullable=False)  # timestamp?
    vid_end: Mapped[DateTime] = mapped_column(DateTime, nullable=False)

    is_downloaded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    download_start: Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    download_end: Mapped[DateTime] = mapped_column(DateTime, nullable=True)

    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    det_start: Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    det_end: Mapped[DateTime] = mapped_column(DateTime, nullable=True)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    @staticmethod
    async def check_if_file_is_downloaded(
        *,
        db_session: AsyncSession,
        file_id: int
    ) -> bool:
        """
        Функция, позволяющая проверить, был ли загружен видеофайл.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД
            file_id (int): ID видеофайла

        Returns:
            is_downloaded (bool): флаг скачивания видеофайла
        """
        file = await db_session.execute(
            select(VideoFile).filter(
                VideoFile.id == file_id
            )
        )
        file = file.scalars().first()
        logger.debug(f'File {file_id} is downloaded: {file.is_downloaded}')

        return file.is_downloaded
    
    @staticmethod
    async def get_unprocessed_files(
        *,
        db_session: AsyncSession
    ) -> list:
        """
        Функция, позволяющая получить список загруженных, но не обработанных видеофайлов.

        Args:
            db_session (AsyncSession): объект асинхронной сессии БД

        Returns:
            files (list): список необработанных видеофайлов
        """
        files = await db_session.execute(
            select(VideoFile).filter(
                VideoFile.is_downloaded == True,
                VideoFile.is_processed == False
            ).order_by(VideoFile.vid_start)
        )
        files = files.scalars().all()

        logger.debug(f'{len(files)} file(s) retrieved for processing: {[file.__dict__ for file in files]}')

        return files
    

# region Проверка окончания обработки файлов за день

class DailyCamCheck(BaseCRUD):
    __tablename__ = "daily_cam_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    camera: Mapped['Camera'] = relationship(back_populates="daily_cam_check")

    date: Mapped[DateTime] = mapped_column(DateTime)
    is_processed: Mapped[bool] = mapped_column(Boolean)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class DailyAllCamCheck(BaseCRUD):
    __tablename__ = "daily_all_cam_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    date: Mapped[DateTime] = mapped_column(DateTime)
    is_processed: Mapped[bool] = mapped_column(Boolean)

    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

# endregion
