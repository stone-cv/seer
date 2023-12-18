import uvicorn
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi import APIRouter

import core.config as cfg
from core.logger import logger
from detector.detector import ObjectDetection
from core.app import app


# app = FastAPI(title="Seer")
# api_router = APIRouter()

def main():
    logger.info('App initiated')

    # async with bot_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    detector = ObjectDetection(capture_index=0)

    # for video in config.videos:
    logger.info('Detection started')
    detector(video_path=cfg.video_path)


if __name__ == '__main__':
    main()
    # uvicorn.run("main:app", port=8000, reload=True)

    # model = YOLO("yolov8n.pt")
    # model.predict(
    #     source="static/stone_4sec.mp4",
    #     show=True,
    #     save=True,
    #     save_txt=True,
    #     device="mps",
    # )
