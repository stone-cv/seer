import asyncio
from datetime import datetime

# TODO: rm
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

import core.config as cfg
from core.logger import logger
from core.app import Application
# from shared_db_models.database import Base
# from shared_db_models.database import db_engine
from shared_db_models.models.models import *
from detection.detector.detector import Detector
from detection.services import process_video_file


async def main():

    logger.info('App initiated')

    """ инициализация моделей для обучения и обработки отдельных видеофайлов """
    # detector = Detector(mode='det')
    # seg_detector = Detector(mode='seg')

    """ инициализация приложения для поиска, скачивания, и обработки видео """
    app = Application()
    app.start()
    try:
        while app.status == 1:
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        app.stop()

    """ создание БД """
    # async with db_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)
    #     logger.info('DB metadata created')

    """ обучение модели """
    # logger.info(f'Training started')

    # await detector.augment_dataset_dir()

    # detector.train_custom(
    #     data='datasets/data.yaml',
    #     split_required=False
    # )
    # logger.info(f'Training complete')

    # detector.augment_dataset_dir()

    """ обработка отдельного видеофайла """

    #await process_video_file(
    #    detector=detector,
    #    seg_detector=seg_detector,
    #    video_path=cfg.video_path,
    #    saw_already_moving=None,
    #    stone_already_present=None,
    #    camera_id=cfg.camera_id
    #)


if __name__ == '__main__':
    asyncio.run(main())
