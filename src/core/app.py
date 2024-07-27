import asyncio
from typing import List
from datetime import datetime
from datetime import timedelta

import core.config as cfg
from core.logger import logger
from core.utils import get_time_from_video_path
from detection.detector.detector import Detector
from detection.services import process_video_file
from downloader.downloader import get_files_list
from downloader.downloader import download_files
from shared_db_models.database import SessionLocal
from shared_db_models.models.models import Event
from shared_db_models.models.models import Camera
from shared_db_models.models.models import VideoFile


class Application:
    def __init__(self):
        self.status: int = 0  # 0 - stopped, 1 - running
        self.detector: Detector = Detector(capture_index=0, mode='det')
        self.detector_seg: Detector = Detector(capture_index=0, mode='seg')
        self.camera_id: int = cfg.camera_id
        self.cam_track_id: int = None  # TODO
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
        self.event_list: List[Event] = []
        self.last_video_end: datetime = datetime.now() - timedelta(hours=self.deep_archive)
        # self.timezone_offset: int = (pytz.timezone(config.get("Application", "timezone", fallback="UTC"))).utcoffset(datetime.now()).seconds
        # logger.info(f"Server offset timezone: {self.__timezone_offset}")

    def start(self):

        # создаем воркеров для поиска видеофайлов
        for _ in range(1):
            task = asyncio.Task(self.search_for_video_files())
            self.tasks_search_video.append(task)

        # создаем воркеров для скачивания видеофайла
        for _ in range(1):
            task = asyncio.Task(self.download_video_files())
            self.tasks_download_video.append(task)

        # создаем воркеров для обработки видеофайла
        for _ in range(1):
            task = asyncio.Task(self.process_video_file())
            self.tasks_process_video.append(task)

        # создаем таску для генерации временных параметров поиска нового видео
        self.task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.last_video_end,
        ))

        self.status = 1

    def stop(self):
        for task in self.tasks_search_video:
            task.cancel()

        for task in self.tasks_download_video:
            task.cancel()

        for task in self.tasks_process_video:
            task.cancel()

        self.task_generate.cancel()

        self.status = 0

    def restart(self):
        for task in self.tasks_search_video:
            task.cancel()

        for task in self.tasks_download_video:
            task.cancel()

        for task in self.tasks_process_video:
            task.cancel()

        self.task_generate.cancel()
        self.queue_search_video: asyncio.Queue = asyncio.Queue()
        self.queue_download_video: asyncio.Queue = asyncio.Queue()
        self.queue_process_video: asyncio.Queue = asyncio.Queue()
        self.task_generate: asyncio.Task = None
        for _ in range(1):
            task = asyncio.Task(self.search_for_video_files())
            self.tasks_search_video.append(task)

        for _ in range(1):
            task = asyncio.Task(self.download_video_files())
            self.tasks_download_video.append(task)

        for _ in range(1):
            task = asyncio.Task(self.process_video_file())
            self.tasks_process_video.append(task)

        self.task_generate = asyncio.Task(self.generate_datetime_queue(
            start_time=self.last_video_end
        ))

    async def generate_datetime_queue(self, start_time: datetime = None):
        """
        Функция, генерирующая временные параметры для поиска файлов на видеорегистраторе

        Args:
            start_time (datetime): начало запрашиваемого периода
        """
        # while True:
        start_time = start_time
        end_time = datetime.now()
        logger.debug(f'start_time: {start_time}; end_time: {end_time}')

        # передаем начало и окончание интервала поиска в очередь для поиска файлов
        await self.queue_search_video.put((start_time, end_time))
        
        # await asyncio.sleep(self.__delay)

    async def search_for_video_files(self):
        """
       Поиск файлов на видеорегистраторе в соответствии с заданным временным интервалом
        """
        while True:

            # извлекаем из очереди время начала и окончания интервала поиска
            start_time, end_time = await self.queue_search_video.get()

            try:
                async with SessionLocal() as session:

                    # получаем track_id камеры
                    if not self.cam_track_id:
                        self.cam_track_id = await Camera.get_track_id_by_camera_id(
                            db_session=session,
                            camera_id=self.camera_id
                        )

                    # получаем список видеофайлов
                    files_dict = await get_files_list(
                        channel=cfg.channel,
                        recorder_ip=cfg.recorder_ip,
                        start_time=start_time,
                        end_time=end_time
                    )

                    # если ни один файл не был найден, через минуту геренируем новый временной интервал
                    if len(files_dict[cfg.channel]) == 0:  # redo
                        logger.info(f"Files not found from {start_time} to {end_time}. Retrying in 60 seconds...")
                        await asyncio.sleep(60)
                        await self.generate_datetime_queue(
                            start_time=self.last_video_end
                        )
                    
                    # если найденное видео начинается раньше, чем заканчивается предыдущий обаботанный файл,
                    # через минуту геренируем новый временной интервал
                    for item in files_dict[cfg.channel]:
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
                        
                        # если найденное видео удовлетворяет всем нашим требованиям,
                        # проверяем, нет ли его в БД (если нет - создаем)
                        else:
                            logger.info(f"Successfully retrived video (start: {vid_start}, end: {vid_end})")

                            video_file = await VideoFile.check_if_exists(
                                db_session=session,
                                param_name='playback_uri',
                                param_val=item['playbackURI']
                            )

                            if not video_file:
                                video_file = await VideoFile.create(
                                    db_session=session,
                                    camera_id=self.camera_id,
                                    path='TBD',
                                    vid_start=vid_start,
                                    vid_end=vid_end,
                                    playback_uri=item['playbackURI']
                                )
                            
                            # складываем данные о файле (+ его ID в БД) в очередь для загрузки видео
                            await self.queue_download_video.put((item, video_file.id))
                            break

            except Exception as exc:
                logger.error(exc)
            finally:
                self.queue_search_video.task_done()

    async def download_video_files(self):
        """
        Скачивание файлов с видеорегистратора
        """
        while True:
            # извлекаем из очереди данные о файле
            video_item, file_id = await self.queue_download_video.get()

            try:
                async with SessionLocal() as session:
                    video_file = await VideoFile.get_by_id(
                        db_session=session,
                        id=file_id
                    )
                    logger.debug(f'File {video_file.id} retrieved for downloading')

                # проверяем, не был ли файл загружен ранее и есть ли в БД запись о его пути
                if video_file.is_downloaded and video_file.path and video_file.path != 'TBD':
                    logger.debug(f'File {video_file.id} already downloaded')
                    filepath = video_file.path

                # если нет - скачиваем
                else:
                    logger.debug(f'Downloading file {video_file.id}...')
                    filepath = await download_files(
                        # channel=cfg.channel,
                        recorder_ip=cfg.recorder_ip,
                        file_id=file_id,
                        data=video_item
                    )

            except Exception as exc:
                logger.error(exc)

            finally:
                logger.debug(f'File {video_file.id} downloaded')
                async with SessionLocal() as session:
                    await VideoFile.update(
                        db_session=session,
                        id=file_id,
                        is_downloaded=True
                    )

                # добавляем путь к файлу (+ его ID в БД) в очередь для обработки видео
                await self.queue_process_video.put((filepath, video_file.id))
                logger.debug(f'{self.queue_process_video.qsize()} files in the queue')

                self.queue_download_video.task_done()
    
    async def process_video_file(self):
        """
        Обработка загруженных видео
        """
        while True:
            # извлекаем из очереди путь к файлу
            filepath, file_id = await self.queue_process_video.get()
            logger.debug(f'File {file_id} ({filepath}) is being processed')
            try:
                det_start = datetime.now()
                logger.debug(f'File {file_id} det_start: {det_start}')

                # обрабатываем видео
                self.saw_already_moving, self.stone_already_present, self.stone_history, self.stone_area_list, self.stone_area, self.event_list = await process_video_file(
                    detector=self.detector,
                    seg_detector=self.detector_seg,
                    video_path=filepath,
                    camera_id=self.camera_id,
                    saw_already_moving = self.saw_already_moving,
                    stone_already_present = self.stone_already_present,
                    stone_history = self.stone_history,
                    stone_area_list = self.stone_area_list,
                    event_list=self.event_list,
                    stone_area = self.stone_area
                )
                logger.debug(f'File {file_id} det_end: {datetime.now()}')

                async with SessionLocal() as session:
                    await VideoFile.update(
                        db_session=session,
                        id=file_id,
                        det_start=det_start,
                        det_end=datetime.now(),
                        is_processed=True
                    )
                
                # получаем время окончания видео из имени файла
                _, self.last_video_end = get_time_from_video_path(filepath)
            except Exception as e:
                print(e)
            finally:
                self.queue_process_video.task_done()
    
            # снова генерим даты для поиска нового видео
            await self.generate_datetime_queue(
                start_time=self.last_video_end
            )
