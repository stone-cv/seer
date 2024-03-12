import asyncio
from datetime import datetime

# TODO: rm
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

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

    """ download files """
    # files_dict = await get_files_list(
    #     channel=cfg.channel,
    #     recorder_ip=cfg.recorder_ip,
    #     start_time=datetime.datetime.fromisoformat("2024-02-14T11:00:00Z".replace("Z", "+03:00")),  # Moscow timezone ?
    #     end_time=datetime.datetime.fromisoformat("2024-02-14T11:59:59Z".replace("Z", "+03:00"))  # Moscow timezone ?
    # )
    # await download_files(
    #     channel=cfg.channel,
    #     recorder_ip=cfg.recorder_ip,
    #     files_dict=files_dict,
    #     queue=queue
    # )

    """ process video & detect objects """
    # logger.info('Detection started')

    # img = cv2.imread('static/vlcsnap-2024-02-12-12h34m02s185.png')
    # results = seg_detector.predict_custom(img)

    # yolo_classes = list(seg_detector.CLASS_NAMES_DICT)
    # classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    # colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # logger.info(f'Results: {results}')
    # for result in results:
    #     for mask, box in zip(result.masks.xy, result.boxes):
    #         points = np.int32([mask])
    #         logger.info(f'Points: {points}')
    #         # cv2.polylines(img, points, True, (255, 0, 0), 1)
    #         color_number = classes_ids.index(int(box.cls[0]))
    #         # cv2.fillPoly(img, points, colors[color_number])  # pizda

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.imwrite('static/vlcsnap-2024-02-12-12h34m02s185_mask.png', img)

    # for r in results:
    #     masks = r.masks.cpu().numpy()
    #     logger.info(masks.__dict__)
    #     for mask in masks:
    #         mask_obj = mask.data[0]
    #         polygon = mask.xy[0].tolist()
    #         logger.info(polygon)

    #         # mask_img = Image.fromarray(mask,"I")

    #         img = Image.open('static/vlcsnap-2024-02-12-12h34m02s185.png')
    #         draw = ImageDraw.Draw(img)
    #         draw.polygon(polygon,outline=(0,255,0), width=5)
    #         img.save('static/1_mask.png')


    await process_video_file(
        detector=detector,
        seg_detector=seg_detector,
        video_path=cfg.video_path,
        saw_already_moving=None,
        stone_already_present=None,
        camera_id=cfg.camera_id
    )
    # await process_live_video(
    #     detector=detector,
    #     camera_id=cfg.camera_id
    # )


if __name__ == '__main__':
    asyncio.run(main())
