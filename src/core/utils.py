import os
import cv2
import uuid
import httpx
from datetime import datetime
from typing import Any
from typing import List

import core.config as cfg
from core.logger import logger
from core.models import Camera
from core.database import SessionLocal


def extract_frame(
        video_path: str,
        camera_roi: List[tuple],
        fps: int = 5
) -> Any:
    """
    Генератор, извлекающий из видео заданное количество кадров в секунду
    и обрезающий кадры под размер области интереса.

    Args:
        video_path (str): путь к видеофайлу
        camera_roi (List[tuple]): координаты области интереса
        fps (int): желаемое количество кадров в секунду
    
    Returns:
        Возвращает кадры с учетом заданного значения FPS.
    """
    video = cv2.VideoCapture(video_path)

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

            # scale_factor = 0.5
            # frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

            # crop frame
            # roi_xmin = int(camera_roi[0][0])
            # roi_ymin = int(camera_roi[0][1])
            # roi_xmax = int(camera_roi[1][0])
            # roi_ymax = int(camera_roi[1][1])
            # cropped_frame = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

            yield frame, frame_idx, video_fps, fps
        frame_idx += 1

    video.release()
    cv2.destroyAllWindows()

    logger.info(f'Extracted {frame_count} frames from {video_path}')


async def crop_images_in_folder(
    folder_path: str
) -> None:
    """
    Функция, позволяющая обрезать изображения в папке

    Args:
        folder_path (str): путь к папке
    
    Returns:
        None
    """
    input_directory = folder_path
    output_directory = f'{folder_path}/cropped_images/'
    os.makedirs(output_directory, exist_ok=True)

    async with SessionLocal() as session:
        camera_roi = await Camera.get_roi_by_camera_id(
            db_session=session,
            camera_id=cfg.camera_id
        )
    roi_xmin = int(camera_roi[0][0])
    roi_ymin = int(camera_roi[0][1])
    roi_xmax = int(camera_roi[1][0])
    roi_ymax = int(camera_roi[1][1])


    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(input_directory, filename))
            cropped_image = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

            output_path = os.path.join(output_directory, 'cropped_' + filename)
            cv2.imwrite(output_path, cropped_image)



def get_time_from_video_path(
        video_path: str
) -> (datetime, datetime):
    """
    Функция, позволяющая извлечь из названия видеофайла время начала и
    окончания видеофрагмента (при условии соблюдения конвенций для наименования файлов:
    <номер канала>_<время начала видеофрагмента>_<время окончания видеофрагмента>.<расширение>).

    Args:
        video_path (str): название видеофайла
    
    Returns:
        start_time (datetime): время начала видео
        end_time (datetime): время окончания видео
    """
    start_time = datetime.fromtimestamp(int(video_path.split('/')[-1].split('_')[1]))
    end_time = datetime.fromtimestamp(int(video_path.split('/')[-1].split('_')[2].split('.')[0]))

    logger.debug(f'Video: {video_path}, start time: {start_time}, end time: {end_time}')
    return start_time, end_time


def xml_helper(
        start_time: datetime,
        end_time: datetime,
        track_id: int
) -> str:
    """ формат lxml файла для передачи параметров поиска файлов"""

    max_result = 1300
    search_position = 0
    search_id = uuid.uuid4()
    metadata = '//recordType.meta.std-cgi.com'

    if isinstance(start_time, (datetime, datetime.date)): # Пока грубая проверка. В следующей версии будет все на Typing и передаваться будет строго datetime.
        start_time = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    if isinstance(end_time, datetime):
        end_time = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    xml_string = f'<?xml version="1.0" encoding="utf-8"?><CMSearchDescription><searchID>{search_id}</searchID>' \
            f'<trackList><trackID>{track_id}</trackID></trackList>' \
            f'<timeSpanList><timeSpan><startTime>{start_time}</startTime>' \
            f'<endTime>{end_time}</endTime></timeSpan></timeSpanList>' \
            f'<maxResults>{max_result}</maxResults>' \
            f'<searchResultPostion>{search_position}</searchResultPostion>' \
            f'<metadataList><metadataDescriptor>{metadata}</metadataDescriptor></metadataList>' \
            f'</CMSearchDescription> '
    logger.debug(f'XML string: {xml_string}')

    return xml_string


async def send_event_json(
    data: str,
    url: str = cfg.json_url,
    auth: str = cfg.json_auth_token
) -> httpx.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url=url, headers=headers, content=data)

            if response.status_code == 200 or response.status_code == 201:
                logger.info(f'JSON POST request sent successfully. Response: {response.text}')
            else:
                logger.error(f'JSON POST request failed.\nResponse status code: {response.status_code}, {response.text}')

    except Exception as exc:
        logger.error(exc)

    # return response


def create_camera_roi(frame) -> list():  # doesn't work here but implemented in "if __name__ == '__main__'"
    """
    Функция, позволяющая обозначить на кадре область интереса и найти ее координаты.

    Args:
        frame: кадр-образец
    
    Returns:
        roi_points (List(tuple)): координаты области интереса
    """
    pass


if __name__ == '__main__':

    # Implementation of create_camera_roi()

    drawing = False
    roi_points = []
    frame = "../static/1.png"

    frame = cv2.imread(frame)

    def draw_roi(event, x, y, flags, param):
        global roi_points, drawing  # eww

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_points = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_points.append((x, y))
            cv2.rectangle(frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

    # Create a window and set the callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_roi)

    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # Reset the ROI
            roi_points = []
            frame = cv2.imread(frame)
        elif key == ord("c"):
            # Confirm the ROI and proceed with further processing
            break

    # logger.info(roi_points)
    print(f'Updated ROI coord: {roi_points}') 
    cv2.destroyAllWindows()

    # async with SessionLocal() as session:
    #     await Camera.update_camera_roi(
    #         db_session=session,
    #         camera_id=1,
    #         roi_coord=str(roi_points)
    #     )
