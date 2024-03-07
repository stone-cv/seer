import asyncio
from typing import List
from datetime import datetime
from datetime import timedelta

from fastapi import FastAPI
from fastapi import APIRouter

import core.config as cfg
from core.logger import logger
from detector.detector import ObjectDetection
from core.downloader import get_files_list
from core.downloader import download_files
from core.scenarios import process_video_file
from core.utils import get_time_from_video_path


# app = FastAPI(title="Seer")
# api_router = APIRouter()


class Application:
    def __init__(self):
        self.status: int = 0  # 0 - stopped, 1 - running
        self.__detector: ObjectDetection = ObjectDetection(capture_index=0)
        self.__camera_id: int = cfg.camera_id
        self.__queue_search_video: asyncio.Queue = asyncio.Queue()
        self.__queue_download_video: asyncio.Queue = asyncio.Queue()
        self.__queue_process_video: asyncio.Queue = asyncio.Queue()
        self.__tasks_search_video: List[asyncio.Task] = []
        self.__tasks_download_video: List[asyncio.Task] = []
        self.__tasks_process_video: List[asyncio.Task] = []
        self.__task_generate: asyncio.Task = None
        self.__delay: int = cfg.delay * 60  # min to sec
        self.__deep_archive: int = cfg.deep_archive
        self.__saw_already_moving: bool = None
        self.__stone_already_present: bool = None
        self.__stone_history: List[bool] = []
        # self.__last_video_end: datetime = datetime.now()-timedelta(minutes=self.__deep_archive)
        self.__last_video_end: datetime = datetime(2024, 3, 7, 11, 18, 45)
        # self.__timezone_offset: int = (pytz.timezone(config.get("Application", "timezone", fallback="UTC"))).utcoffset(datetime.now()).seconds
        # logger.info(f"Server offset timezone: {self.__timezone_offset}")

    def start(self):
        # генерируем воркеров для поиска видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__search_for_video_files())
            self.__tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__download_video_files())
            self.__tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__process_video_file())
            self.__tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.__task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.__last_video_end,
        ))

        # устанавливаем статус апликахе
        self.status = 1

    def stop(self):
        # останавливаем таски поиска видео файлов
        for task in self.__tasks_search_video:
            task.cancel()

        # останавливаем таски скачивания видео файлов
        for task in self.__tasks_download_video:
            task.cancel()

        # останавливаем таски обработки видео файлов
        for task in self.__tasks_process_video:
            task.cancel()

        # отсанавливаем таск для формирования очереди данных
        self.__task_generate.cancel()

        # устанавливаем статус аппликахе
        self.status = 0

    def restart(self):
        # останавливаем таски поиска видео файлов
        for task in self.__tasks_search_video:
            task.cancel()

        # останавливаем таски скачивания видео файлов
        for task in self.__tasks_download_video:
            task.cancel()

        # останавливаем таски обработки видео файлов
        for task in self.__tasks_process_video:
            task.cancel()

        # отсанавливаем таск для формирования очереди данных
        self.__task_generate.cancel()
        self.__queue_search_video: asyncio.Queue = asyncio.Queue()
        self.__queue_download_video: asyncio.Queue = asyncio.Queue()
        self.__queue_process_video: asyncio.Queue = asyncio.Queue()
        self.__task_generate: asyncio.Task = None
        for _ in range(1):
            task = asyncio.Task(self.__search_for_video_files())
            self.__tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__download_video_files())
            self.__tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__process_video_file())
            self.__tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.__task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.__last_video_end
        ))

    async def generate_datetime_queue(self, start_time: datetime = None):
        """
        ???
        """
        # while True:
        start_time = start_time
        end_time = datetime.now()
        logger.debug(f'start_time: {start_time}; end_time: {end_time}')

        await self.__queue_search_video.put((start_time, end_time))
        
        # await asyncio.sleep(self.__delay)

    async def __search_for_video_files(self):
        """
        ???
        """
        while True:
            start_time, end_time = await self.__queue_search_video.get()
            try:
                files_dict = await get_files_list(
                    channel=cfg.channel,
                    recorder_ip=cfg.recorder_ip,
                    start_time=start_time,
                    end_time=end_time
                )

                if len(files_dict[cfg.channel]) == 0:  # redo
                    logger.info(f"Files not found from {start_time} to {end_time}. Retrying in 60 seconds...")
                    await asyncio.sleep(60)
                    await self.generate_datetime_queue(
                        start_time=self.__last_video_end
                    )
                
                # check if we have already downloaded the file for this time period
                for item in files_dict[cfg.channel]:  # redo
                    vid_start = datetime.strptime(item['startTime'], "%Y-%m-%dT%H:%M:%SZ")
                    vid_end = datetime.strptime(item['endTime'], "%Y-%m-%dT%H:%M:%SZ")
                    if vid_start < self.__last_video_end:
                        if len(files_dict[cfg.channel]) > 1:
                            continue
                        logger.info(f"Video starts ({vid_start}) earlier than the last video ends ({self.__last_video_end}). Retrying in 60 seconds...")
                        await asyncio.sleep(60)
                        await self.generate_datetime_queue(
                            start_time=self.__last_video_end
                        )
                    else:
                        logger.info(f"Successfully retrived video file (start: {vid_start}, end: {vid_end})")
                        await self.__queue_download_video.put(item)

            except Exception as exc:
                logger.error(exc)
            finally:
                self.__queue_search_video.task_done()

    async def __download_video_files(self):
        """
        ???
        """
        while True:
            data = await self.__queue_download_video.get()
            try:
                filepath = await download_files(
                    channel=cfg.channel,
                    recorder_ip=cfg.recorder_ip,
                    data=data
                )
                await self.__queue_process_video.put(filepath)

            except Exception as exc:
                logger.error(exc)
            finally:
                self.__queue_download_video.task_done()
    
    async def __process_video_file(self):
        """
        ???
        """
        while True:
            item = await self.__queue_process_video.get()
            try:
                self.__saw_already_moving, self.__stone_already_present, self.__stone_history = await process_video_file(
                    detector=self.__detector,
                    video_path=item,
                    camera_id=self.__camera_id,
                    saw_already_moving = self.__saw_already_moving,
                    stone_already_present = self.__stone_already_present,
                    stone_history = self.__stone_history
                )
                _, self.__last_video_end = get_time_from_video_path(item)
            except Exception as e:
                print(e)
            finally:
                self.__queue_process_video.task_done()
        
                # снова генерим даты для нового видео
                await self.generate_datetime_queue(
                    start_time=self.__last_video_end
                )
