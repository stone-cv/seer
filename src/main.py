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
from core.database import Base
from core.database import db_engine
from detector.detector import Detector
from core.downloader import get_files_list
from core.downloader import download_files
from core.scenarios import process_video_file
from core.scenarios import process_live_video
from core.scenarios import calculate_segment_area
from core.utils import crop_images_in_folder
from core.models import *
from core.app import Application


async def main():

    logger.info('App initiated')

    detector = Detector(mode='det')
    seg_detector = Detector(mode='seg')

    """ init application """
    # app = Application()
    # app.start()
    # try:
    #     while app.status == 1:
    #         await asyncio.sleep(5)
    # except KeyboardInterrupt:
    #     app.stop()

    """ create db """
    # async with db_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)
    #     logger.info('DB metadata created')

    """ train model """
    # logger.info(f'Training started')
    # detector.train_custom(
    #     data='datasets/data.yaml',
    #     split_required=False
    # )
    # logger.info(f'Training complete')

    # detector.augment_dataset_dir()

    """ process video & detect objects """

    # region segmentation

    # img = '/Users/macbook/Pictures/vlcsnap-2024-03-22-19h54m51s252.png'
    # results = seg_detector.model(img)
    # segment = seg_detector.parse_segmentation(results)
    # seg_detector.plot_segmentation(segment, img)
    # calculate_segment_area(segment)

    # endregion


    await process_video_file(
        detector=detector,
        seg_detector=seg_detector,
        video_path=cfg.video_path,
        saw_already_moving=None,
        stone_already_present=None,
        camera_id=cfg.camera_id
    )


if __name__ == '__main__':
    asyncio.run(main())
