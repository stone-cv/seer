import asyncio

import core.config as cfg
from core.logger import logger
from core.database import Base
from core.database import db_engine
from detector.detector import ObjectDetection
from core.app import process_video_file
from core.app import process_live_video
from core.models import *


async def main():
    logger.info('App initiated')

    # async with db_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)
    #     await conn.run_sync(Base.metadata.create_all)
    #     logger.info('DB metadata created')

    detector = ObjectDetection(capture_index=0)  # here? source?

    #train
    # logger.info(f'Training started')
    # detector.train_custom(data='datasets/data.yaml')
    # logger.info(f'Training complete')

    # for video in config.videos:
    logger.info('Detection started')
    # await process_video_file(
    #     detector=detector,
    #     video_path=cfg.video_path,
    #     camera_id=1  # deafult for now
    # )
    await process_live_video(
        detector=detector,
        camera_id=1  # deafult for now
    )


if __name__ == '__main__':
    asyncio.run(main())
    # uvicorn.run("main:app", port=8000, reload=True)
