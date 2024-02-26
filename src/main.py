import asyncio
from datetime import datetime

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

    detector = ObjectDetection(capture_index=0)  # here? source?

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
    logger.info('Detection started')

    await process_video_file(
        detector=detector,
        video_path=cfg.video_path,
        stone_already_present=True,  # remove
        camera_id=1  # default for now
    )
    # await process_live_video(
    #     detector=detector,
    #     camera_id=1  # default for now
    # )


if __name__ == '__main__':
    asyncio.run(main())
