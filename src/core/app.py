import asyncio

from datetime import datetime
from datetime import timedelta
from typing import List

from fastapi import FastAPI
from fastapi import APIRouter

import core.config as cfg
from core.logger import logger
# from tracker.tracker import Sort
from detector.detector import ObjectDetection
from core.models import Camera
from core.database import SessionLocal
from core.utils import extract_frame
from core.utils import get_time_from_video_path
from core.scenarios import is_in_roi
from core.scenarios import check_for_motion
from core.scenarios import check_if_object_present
from core.downloader import get_files_list
from core.downloader import download_files
from core.scenarios import process_video_file


app = FastAPI(title="Seer")
api_router = APIRouter()


class Application:
    def __init__(self):
        self.status: int = 0
        # self.__url: str = config.get('Application', 'URL')
        # self.__login: str = config.get('Application', 'login')
        # self.__password: str = config.get("Application", "password")
        self.__detector: ObjectDetection = ObjectDetection(capture_index=0)
        self.__queue_search_video: asyncio.Queue = asyncio.Queue()
        self.__queue_download_video: asyncio.Queue = asyncio.Queue()
        self.__queue_process_video: asyncio.Queue = asyncio.Queue()
        self.__tasks_search_video: List[asyncio.Task] = []
        self.__tasks_download_video: List[asyncio.Task] = []
        self.__tasks_process_video: List[asyncio.Task] = []
        self.__task_generate: asyncio.Task = None
        self.__delay: int = cfg.delay * 60  # min to sec
        self.__deep_archive: int = cfg.deep_archive
        # self.__timezone_offset: int = (pytz.timezone(config.get("Application", "timezone", fallback="UTC"))).utcoffset(datetime.now()).seconds
        # logger.info(f"Server offset timezone: {self.__timezone_offset}")

    def start(self):
        # генерируем воркеров для поиска видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__search_video_files())
            self.__tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__download_video_files())
            self.__tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__process_file())
            self.__tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.__task_generate = asyncio.Task(self.generate_datetime_queue())

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
            task = asyncio.Task(self.__search_video_files())
            self.__tasks_search_video.append(task)

        # генерируем воркеров для скачивания видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__download_video_files())
            self.__tasks_download_video.append(task)

        # генерируем воркеров для обработки видео файлов
        for _ in range(1):
            task = asyncio.Task(self.__process_file())
            self.__tasks_process_video.append(task)

        # генерируем таск для получения вставки дат и времени в очередь
        self.__task_generate = asyncio.Task(self.generate_datetime_queue())

    async def generate_datetime_queue(self):
        while True:
            end_time = datetime.now()  # TODO check which is end and start
            start_time = end_time - timedelta(minutes=self.__deep_archive)
            logger.debug(f'start_time: {start_time}; end_time: {end_time}')

            await self.__queue_search_video.put((start_time, end_time))

            # while end_time <= start_time:
            #     logger.debug(f'start_time: {start_time}; end_time: {end_time}')
            #     await self.__queue_search_video.put((start_time, end_time))
            #     end_time += timedelta(days=1)
            # while self.__queue_search_video.qsize() != 0 or self.__queue_download_video.qsize() != 0 or self.__queue_process_video.qsize() != 0:
            #     logger.info(f"Queue not empty; Queue to get data: {self.__queue_search_video.qsize()}; Queue to parse data: {self.__queue_download_video.qsize()}; Queue to save data: {self.__queue_process_video.qsize()}")
            #     await asyncio.sleep(5)
            
            await asyncio.sleep(self.__delay)

    async def __search_video_files(self):
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
                await self.__queue_download_video.put(files_dict)  # TODO each file separate
                logger.info(f"Successfully retrived files from {start_time} and {end_time}")

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
                for item in data[cfg.channel]:  # redo
                    filepath = await download_files(
                        channel=cfg.channel,
                        recorder_ip=cfg.recorder_ip,
                        files_dict=item
                    )
                    await self.__queue_process_video.put(filepath)

            except Exception as exc:
                logger.error(exc)
            finally:
                self.__queue_download_video.task_done()
    
    async def __process_file(self):
        """
        ???
        """
        while True:
            item = await self.__queue_process_video.get()
            try:
                await process_video_file(
                    detector=self.__detector,
                    video_path=item,
                    camera_id=1  # deafult for now
                )
            except Exception as e:
                print(e)
            finally:
                self.__queue_process_video.task_done()


async def main():
    logger.info("application start")
    app = Application()
    # await database.connect()
    app.start()
    try:
        while app.status == 1:
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        app.stop()


if __name__ == '__main__':
    asyncio.run(main())
