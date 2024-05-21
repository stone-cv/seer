import asyncio
from typing import List
from datetime import datetime
from datetime import timedelta

from fastapi import FastAPI
from fastapi import APIRouter

import core.config as cfg
from core.logger import logger
from detector.detector import Detector
from core.downloader import get_files_list
from core.downloader import download_files
from core.scenarios import process_video_file
from core.utils import get_time_from_video_path


class Application:
    def __init__(self):
        self.status: int = 0  # 0 - stopped, 1 - running
        self.detector: Detector = Detector(capture_index=0, mode='det')
        self.detector_seg: Detector = Detector(capture_index=0, mode='seg')
        self.camera_id: int = cfg.camera_id
        self.queue_search_video: asyncio.Queue = asyncio.Queue()
        self.queue_download_video: asyncio.Queue = asyncio.Queue()
        self.queue_process_video: asyncio.Queue = asyncio.Queue()
        self.tasks_search_video: List[asyncio.Task] = []
        self.tasks_download_video: List[asyncio.Task] = []
        self.tasks_process_video: List[asyncio.Task] = []
        self.task_generate: asyncio.Task = None
        self.delay: int = cfg.delay * 60  # min to sec
        self.deep_archive: int = cfg.deep_archive
        self.saw_already_moving: bool = None
        self.stone_already_present: bool = None
        self.stone_history: List[bool] = []
        self.stone_area_list: List[float] = []
        self.stone_area: float = 0
        self.last_video_end: datetime = datetime(2024, 5, 21, 15, 50, 0)
        # self.timezone_offset: int = (pytz.timezone(config.get("Application", "timezone", fallback="UTC"))).utcoffset(datetime.now()).seconds
        # logger.info(f"Server offset timezone: {self.__timezone_offset}")

    def start(self):
        # генерируем воркеров для поиска видео файлов
        for _ in range(1):
            task = asyncio.Task(self.search_for_video_files())
            self.tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.download_video_files())
            self.tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.process_video_file())
            self.tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.last_video_end,
        ))

        # устанавливаем статус апликахе
        self.status = 1

    def stop(self):
        # останавливаем таски поиска видео файлов
        for task in self.tasks_search_video:
            task.cancel()

        # останавливаем таски скачивания видео файлов
        for task in self.tasks_download_video:
            task.cancel()

        # останавливаем таски обработки видео файлов
        for task in self.tasks_process_video:
            task.cancel()

        # отсанавливаем таск для формирования очереди данных
        self.task_generate.cancel()

        # устанавливаем статус аппликахе
        self.status = 0

    def restart(self):
        # останавливаем таски поиска видео файлов
        for task in self.tasks_search_video:
            task.cancel()

        # останавливаем таски скачивания видео файлов
        for task in self.tasks_download_video:
            task.cancel()

        # останавливаем таски обработки видео файлов
        for task in self.tasks_process_video:
            task.cancel()

        # отсанавливаем таск для формирования очереди данных
        self.task_generate.cancel()
        self.queue_search_video: asyncio.Queue = asyncio.Queue()
        self.queue_download_video: asyncio.Queue = asyncio.Queue()
        self.queue_process_video: asyncio.Queue = asyncio.Queue()
        self.task_generate: asyncio.Task = None
        for _ in range(1):
            task = asyncio.Task(self.search_for_video_files())
            self.tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.download_video_files())
            self.tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.process_video_file())
            self.tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.last_video_end
        ))

    async def generate_datetime_queue(self, start_time: datetime = None):
        """
        ???
        """
        # while True:
        start_time = start_time
        end_time = datetime.now()
        logger.debug(f'start_time: {start_time}; end_time: {end_time}')

        await self.queue_search_video.put((start_time, end_time))
        
        # await asyncio.sleep(self.__delay)

    async def search_for_video_files(self):
        """
        ???
        """
        while True:
            start_time, end_time = await self.queue_search_video.get()
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
                        start_time=self.last_video_end
                    )
                
                # check if we have already downloaded the file for this time period
                for item in files_dict[cfg.channel]:  # redo
                    vid_start = datetime.strptime(item['startTime'], "%Y-%m-%dT%H:%M:%SZ")
                    vid_end = datetime.strptime(item['endTime'], "%Y-%m-%dT%H:%M:%SZ")
                    if vid_start < self.last_video_end:
                        if len(files_dict[cfg.channel]) > 1:
                            continue
                        logger.info(f"Video starts ({vid_start}) earlier than the last video ends ({self.last_video_end}). Retrying in 60 seconds...")
                        await asyncio.sleep(60)
                        await self.generate_datetime_queue(
                            start_time=self.last_video_end
                        )
                    else:
                        logger.info(f"Successfully retrived video (start: {vid_start}, end: {vid_end})")
                        await self.queue_download_video.put(item)

            except Exception as exc:
                logger.error(exc)
            finally:
                self.queue_search_video.task_done()

    async def download_video_files(self):
        """
        ???
        """
        while True:
            data = await self.queue_download_video.get()
            try:
                filepath = await download_files(
                    channel=cfg.channel,
                    recorder_ip=cfg.recorder_ip,
                    data=data
                )
                await self.queue_process_video.put(filepath)

            except Exception as exc:
                logger.error(exc)
            finally:
                self.queue_download_video.task_done()
    
    async def process_video_file(self):
        """
        ???
        """
        while True:
            item = await self.queue_process_video.get()
            try:
                # self.saw_already_moving, self.stone_already_present, self.stone_history, self.stone_area_list, self.stone_area = await process_video_file(
                #     detector=self.detector,
                #     seg_detector=self.detector_seg,
                #     video_path=item,
                #     camera_id=self.camera_id,
                #     saw_already_moving = self.saw_already_moving,
                #     stone_already_present = self.stone_already_present,
                #     stone_history = self.stone_history,
                #     stone_area_list = self.stone_area_list,
                #     stone_area = self.stone_area
                # )
                _, self.last_video_end = get_time_from_video_path(item)
            except Exception as e:
                print(e)
            finally:
                self.queue_process_video.task_done()
    
            # снова генерим даты для нового видео
            await self.generate_datetime_queue(
                start_time=self.last_video_end
            )
