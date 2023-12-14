import cv2
from typing import Any
from datetime import datetime

from logger import logger


def extract_frame(video_path: str, fps: int = 5) -> Any:
    video = cv2.VideoCapture(video_path)

    # video_start_time = video.get(cv2.CAP_PROP_CREATION_TIME)
    # logger.debug(f'Video start time: {video_start_time}')

    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    logger.debug(f'Video path: {video_path}, FPS: {video_fps}')

    frame_interval = int(video_fps / fps)
    frame_idx = 0
    frame_count = 0

    while frame_count <= video_frame_count:
        ret, frame = video.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_count += 1
            yield frame, frame_idx, video_fps
        frame_idx += 1

    video.release()
    cv2.destroyAllWindows()

    logger.info(f'Extracted {frame_count} frames from {video_path}')
    return


def get_time_from_video_path(video_path: str):
    start_time = datetime.fromtimestamp(int(video_path.split('/')[-1].split('_')[1]))
    end_time = datetime.fromtimestamp(int(video_path.split('/')[-1].split('_')[2].split('.')[0]))

    # start_time = datetime.fromtimestamp(start_time_unix)
    # end_time = datetime.fromtimestamp(end_time_unix)

    logger.debug(f'Video: {video_path}, start time: {start_time}, end time: {end_time}')
    return start_time, end_time
