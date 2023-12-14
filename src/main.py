from ultralytics import YOLO

import config
from logger import logger
from detector import ObjectDetection


def main():
    logger.info('App initiated')

    logger.info('Detection started')
    detector = ObjectDetection(capture_index=0)

    # for video in config.videos:
    detector(video_path=config.video_path)


if __name__ == '__main__':
    main()

    # model = YOLO("yolov8n.pt")
    # model.predict(
    #     source="static/stone_4sec.mp4",
    #     show=True,
    #     save=True,
    #     save_txt=True,
    #     device="mps",
    # )
