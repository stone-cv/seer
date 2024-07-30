import os
import re
import cv2
import uuid
import json
import httpx
import traceback
import numpy as np
from datetime import datetime
from typing import Any
from typing import List

import core.config as cfg
from core.logger import logger
from shared_db_models.models.models import Camera
from shared_db_models.database import SessionLocal


def extract_frame(
        video_path: str,
        camera_roi: List[tuple],
        fps: int = 5
) -> Any:
    """
    Генератор, извлекающий из видео заданное количество кадров в секунду
    и обрабатывающий их в целях улучшения изображения для последующей детекции.

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

    # извлечение производится, пока не дойдем до конца видео
    while frame_count <= video_frame_count:
        ret, frame = video.read()
        if not ret:
            break

        # извлекается каждый Х кадр (если индекс кадра делится без остатка на заданный интервал)
        if frame_idx % frame_interval == 0:
            frame_count += 1

            # дополнительная настройка яркости и контраста
            brightness = 0
            contrast = 1.2
            gamma = 3
            frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), gamma, brightness)

            # # Sharpen the image 
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # frame = cv2.filter2D(frame, -1, kernel)

            # # adjust color
            # # Convert the image from BGR to HSV color space 
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) 
            # # Adjusts the hue by multiplying it by 0.7 
            # frame[:, :, 0] = frame[:, :, 0] * 1
            # # Adjusts the saturation by multiplying it by 1.5 
            # frame[:, :, 1] = frame[:, :, 1] * 1.5  # ! 1.5
            # # Adjusts the value by multiplying it by 0.5 
            # frame[:, :, 2] = frame[:, :, 2] * 1
            # # Convert the image back to BGR color space 
            # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) 

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
) -> tuple[datetime, datetime]:
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

    return (start_time, end_time)


def xml_helper(
        start_time: datetime,
        end_time: datetime,
        track_id: int
) -> str:
    """
    Форматирование lxml файла для передачи параметров поиска файлов в API видеорегистратора

    Args:
        start_time (datetime): время начала поиска
        end_time (datetime): время окончания поиска
        track_id (int): идентификатор камеры для регистратора
    
    Returns:
        xml_string (str): отформатированная строка
    """

    max_result = 1300
    search_position = 0
    search_id = uuid.uuid4()
    metadata = '//recordType.meta.std-cgi.com'

    if isinstance(start_time, (datetime, datetime.date)): # TODO Typing & datetime
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


async def save_frame_img(
    frame: np.ndarray,
    detection_time: datetime
) -> str:
    """
    Сохранение кадра в файл (изображение)

    Args:
        frame (np.ndarray): кадр
        detection_time (datetime): время на кадре
    
    Returns:
        filename (str): имя сохраненного файла
    """

    filename = f'screenshot_{detection_time.strftime("%m-%d-%Y_%H-%M-%S")}.png'
    saved_img = cv2.imwrite(filename, frame)
    logger.debug(f"\n\n\nImage saved: {saved_img} ({filename})\n\n\n")

    return filename


async def send_event_info(
    data: str,
    frame: np.ndarray,
    detection_time: datetime,
    url_json: str = cfg.json_url,
    url_img: str = cfg.img_url,
    auth: str = cfg.json_auth_token
) -> None:
    """
    Функция, отправляющие данные о созданном событие на внешний сервер:
        - отправляет POST-запрос с данными о событии в формате JSON;
        - извлекает из ответа ID созданной записи;
        - отправляет POST-запрос со скриншотом кадра.

    Args:
        data (str): JSON с данными о событии
        frame (np.ndarray): кадр из видео
        detection_time (datetime): время на кадре
        url_json (str): URL для отправки POST-запроса с данными о событии
        url_img (str): URL для отправки POST-запроса со скриншотом кадра
        auth (str): авторизационный токен
    
    Returns:
        None
    """

    headers_json = {
        "Content-Type": "application/json",
        "Authorization": auth
    }

    headers_img = {
        "Authorization": auth
    }

    try:
        async with httpx.AsyncClient() as client:
            timeout = httpx.Timeout(10.0, read=None)
            print('JSON POST request sent...')

            # отправляем POST-запрос с данными о событии
            response = await client.post(url=url_json, headers=headers_json, content=data, timeout=timeout)

            if response.status_code == 200 or response.status_code == 201:
                logger.info(f'JSON POST request sent successfully. Response: {response.text}.')

                # извлекаем из ответа ID созданной записи
                resp_id = re.search('"id":([0-9]+)', response.text).group(1)
                logger.debug(f'JSON response ID: {resp_id}')

                if cfg.send_img:
                    # сохраняем файл со скриншотом кадра
                    filename = await save_frame_img(frame=frame, detection_time=detection_time)
                    data = {
                        'holderType': 'manufacturingOperation',
                        'manufacturingOperationId': resp_id
                    }
                    files = {
                        'file': (filename, open(filename, 'rb'), 'application/octet-stream'),
                        'formData': (None, json.dumps(data), 'application/json')
                    }

                    # отправляем POST-запрос со скриншотом кадра
                    r = await client.post(url=url_img, headers=headers_img, files=files)

                    if response.status_code == 200 or response.status_code == 201:
                        logger.info(f'Image POST request sent successfully. Response: {r.text}.')

                        # удаляем созданный файл
                        os.remove(filename)
                    else:
                        logger.error(f'Image POST request failed.\nResponse status code: {response.status_code}, {response.text}')

            else:
                logger.error(f'JSON POST request failed.\nResponse status code: {response.status_code}, {response.text}')

    except Exception as exc:
        logger.error(f'{exc} {traceback.format_exc()}')

    # return response


def create_camera_roi(
    frame: np.ndarray
) -> None:  # doesn't work here but implemented in "if __name__ == '__main__'"
    """
    Функция, позволяющая обозначить на кадре область интереса (ROI) и найти ее координаты.
    Исполнятеся при вызове с текущей страницы

    Args:
        frame (np.ndarray): кадр-образец
    
    Returns:
        Координаты ROI выводятся в терминале
    """
    pass


if __name__ == '__main__':

    # Implementation of create_camera_roi()

    drawing = False
    roi_points = []
    frame = "video/1.png"  # путь к кадру-образцу

    frame = cv2.imread(frame)

    # отрисовка ROI на кадре (нажатием левой кнопки мыши задается верхняя левая и нижняя правая точки)
    def draw_roi(event, x, y, flags, param):
        global roi_points, drawing  # redo

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_points = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_points.append((x, y))
            cv2.rectangle(frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_roi)

    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # сброс заданной ROI
            roi_points = []
            frame = cv2.imread(frame)
        elif key == ord("c"):
            # подтвержение ROI и вывод координат
            break

    # вывод координат ROI в терминал
    print(f'Updated ROI coord: {roi_points}') 
    cv2.destroyAllWindows()
