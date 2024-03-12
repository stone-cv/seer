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
from detector.detector import ObjectDetection
from core.downloader import get_files_list
from core.downloader import download_files
from core.scenarios import process_video_file
from core.scenarios import process_live_video
from core.utils import crop_images_in_folder
from core.models import *
from core.app import Application


async def main():

    logger.info('App initiated')

    """ init application """
    # app = Application()
    # app.start()
    # try:
    #     while app.status == 1:
    #         await asyncio.sleep(5)
    # except KeyboardInterrupt:
    #     app.stop()

    detector = ObjectDetection(capture_index=0, mode='det')
    seg_detector = ObjectDetection(capture_index=0, mode='seg')

    """ create db """
    # async with db_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)
    #     logger.info('DB metadata created')

    """ train model """
    # logger.info(f'Training started')
    # detector.train_custom(
    #     data='datasets/data.yaml',
    #     split_required=True
    # )
    # logger.info(f'Training complete')

    """ process video & detect objects """
    # logger.info('Detection started')

    # img = cv2.imread('static/vlcsnap-2024-02-12-12h34m02s185.png')
    img = cv2.imread('static/1.png')

    # results = seg_detector.model(
    #     # 'static/test-start-sawing-short_1702639705_1702639796.mp4',
    #     'static/1.png',
    #     # stream=True, 
    #     show=True
    # )
    results = seg_detector.model(img)

    # yolo_classes = list(seg_detector.model.names.values())  # seg_detector.CLASS_NAMES_DICT
    # classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    # colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # logger.info(f'Results: {results}')
    for result in results:
        # for mask, box in zip(result.masks.xy, result.boxes):
        for mask in result.masks.xy:
            logger.debug(f'Mask coords: {mask}')

            # calculate area
            ratio_px_to_cm = 1.2  # 0.8 recalculate

            # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            area_px = cv2.contourArea(mask)
            logger.info(f'Contour area: {area_px} px')
            area_cm2 = round(area_px / (ratio_px_to_cm ** 2))
            logger.info(f'Slab area: {area_cm2} cm^2')

            cv2.polylines(img, np.int32([mask]), True, (255, 0, 0), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

    #         color_number = classes_ids.index(int(box.cls[0]))

    #         # cv2.fillPoly(img, points, colors[color_number])  # crashes
    #         # Create a mask image
    #         mask_img = np.zeros_like(img)
    #         cv2.drawContours(mask_img, [points], 0, colors[color_number], -1)

    #         # Apply the mask to the original image
    #         img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.imwrite('static/1_mask.png', img)


    # await process_video_file(
    #     detector=detector,
    #     seg_detector=seg_detector,
    #     video_path=cfg.video_path,
    #     saw_already_moving=None,
    #     stone_already_present=None,
    #     camera_id=cfg.camera_id
    # )


if __name__ == '__main__':
    asyncio.run(main())
