import asyncio
from datetime import datetime

import core.config as cfg
from core.logger import logger
from core.database import Base
from core.database import db_engine
from detector.detector import ObjectDetection
from core.downloader import get_files_list
from core.downloader import download_files
from core.app import process_video_file
from core.app import process_live_video
from core.models import *


async def main():

    logger.info('App initiated')
    detector = ObjectDetection(capture_index=0)  # here? source?

    queue = asyncio.Queue()

    """ create db """
    # async with db_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)
    #     logger.info('DB metadata created')

    """ train model """
    logger.info(f'Training started')
    detector.train_custom(
        data='datasets/data.yaml',
        split_required=False
    )
    logger.info(f'Training complete')

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

    # while not queue.empty():
    #     logger.debug(f'Processing video queue size: {queue.qsize()}')
    #     video_file = await queue.get()
    #     await process_video_file(
    #         detector=detector,
    #         video_path=video_file,
    #         camera_id=1  # deafult for now
    #     )
    # await process_live_video(
    #     detector=detector,
    #     camera_id=1  # deafult for now
    # )


if __name__ == '__main__':
    asyncio.run(main())
    # uvicorn.run("main:app", port=8000, reload=True)
