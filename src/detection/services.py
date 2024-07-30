import cv2
import asyncio
import numpy as np

from typing import List
from datetime import datetime
from datetime import timedelta
from threading import Thread
from sqlalchemy.ext.asyncio import AsyncSession

import core.config as cfg
from core.logger import logger
from core.utils import extract_frame
from core.utils import send_event_info
from core.utils import get_time_from_video_path
from detection.detector.detector import Detector
from shared_db_models.models.models import Event
from shared_db_models.models.models import Camera
from shared_db_models.database import SessionLocal


async def process_video_file(
    detector: Detector,
    seg_detector: Detector,
    video_path: str,
    camera_id: int,
    saw_already_moving: bool = False,
    stone_already_present: bool = False,
    stone_history: List[bool] = [],
    stone_area_list: List[float] = [],
    stone_area: float = 0,
    event_list: List[Event] = [],
) -> tuple:
    """
    Функция, объединяющая всю логику обработки видеофайла:
        - извлечение кадров из видеофайла;
        - детекция объектов с помощью моделей (+ при необходимости сегментация);
        - парсинг результатов детекции;
        - выполнение логики срабатывания событий;
        - отправка информации об событиях.

    Args:
        detector (Detector): модель, выполяющая детекцию объектов
        seg_detector (Detector): модель, выполняющая сегментацию
        video_path (str): путь к видеофайлу
        camera_id (int): ID камеры
        saw_already_moving (bool): флаг, демонстрирующий движение пилы. По умолчанию - False
        stone_already_present (bool): флаг, демонстрирующий наличие камня. По умолчанию - False
        stone_history (List[bool]): история наличия камня за заданный промежуток
        stone_area_list (List[float]): список площадей камня за заданный промежуток
        stone_area (float): площадь камня в см2
        event_list (List[Event]): список созданных событий
    
    Returns:
        tuple: кортеж, содержащий обновленные значения следующих переменных:
            - saw_already_moving (bool)
            - stone_already_present (bool)
            - stone_history (List[bool])
            - stone_area_list (List[float])
            - stone_area (float)
            - event_list (List[Event])
    """

    vid_start_time, _ = get_time_from_video_path(video_path)

    # инициализация переменных для логики пилы
    saw_xywh_history = []
    saw_track_magn = 0

    # инициализация переменных для логики камней
    class_ids = []
    forklift_history = []
    
    try:
        async with SessionLocal() as session:
            camera_roi = await Camera.get_roi_by_camera_id(  # move up
                db_session=session,
                camera_id=camera_id
            )

            # извлечение кадров из видеофайла
            frame_generator = extract_frame(
                video_path=video_path,
                camera_roi=camera_roi,
                fps=cfg.required_fps
            )

            for frame, frame_idx, video_fps, curr_fps in frame_generator:
                logger.debug(f'Frame ID: {frame_idx}')

                detection_time = vid_start_time + timedelta(seconds=frame_idx/video_fps)
                logger.debug(f'Detection time: {detection_time}')

                # обработка кадра с помощью модели и трекера
                results = detector.track_custom(source=frame)

                for result in results:
                    frame_pred = detector.parse_detections(result)

                    for item in frame_pred:
                        item['time'] = detection_time

                        # проверка, находится ли объект в интресующей нас области (ROI)
                        item_in_roi = await is_in_roi(
                            roi_xyxy=camera_roi,
                            object_xyxy=item['xyxy']
                        )
                        if item_in_roi:
                            class_ids.append(item['class_id'])

                            # логика, относящаяся к пиле: проверка на движение
                            if item['class_id'] == detector.class_ids_dict['saw']:
                                saw_track_magn, saw_already_moving, saw_event = await check_for_motion(
                                    db_session=session,
                                    xywh_history=saw_xywh_history,
                                    detected_item=item,
                                    saw_track_magn=saw_track_magn,
                                    already_moving=saw_already_moving,
                                    curr_fps=curr_fps,
                                    detection_time=detection_time,
                                    camera_id=camera_id,
                                    # frame=frame
                                )
                                if saw_event:
                                    event_list.append(saw_event)

                    # логика, относящаяся к камням: проверка на наличие и перемещение
                    stone_already_present, stone_history, stone_event = await check_if_stone_present_or_transferred(
                        db_session=session,
                        stone_id=detector.class_ids_dict['stone'],
                        detected_class_ids=class_ids,
                        object_history=stone_history,
                        object_already_present=stone_already_present,
                        forklift_id=detector.class_ids_dict['forklift'],
                        forklift_history=forklift_history,
                        saw_already_moving=saw_already_moving,
                        curr_fps=curr_fps,
                        detection_time=detection_time,
                        camera_id=camera_id,
                    )
                    if stone_event:  #and stone_event.type_id != 2:
                        event_list.append(stone_event)

                    if len(event_list) > 0:
                        logger.debug(f'Event list: {event_list}')

                        # вычисление площади камня
                        stone_area, stone_area_list = await get_stone_area(
                            db_session=session,
                            frame=frame,
                            seg_detector=seg_detector,
                            stone_already_present=stone_already_present,
                            stone_area_list=stone_area_list,
                            stone_area=stone_area,
                            saw_already_moving=saw_already_moving,
                            curr_fps=curr_fps,
                            camera_id=camera_id,
                            detection_time=detection_time
                        )

                        logger.debug(f'Stone area: {stone_area}')

                        # если площадь < 0: она еще не была рассчитана достаточное кол-во раз для вычсления среднего значения
                        # если 0 < площадь < 1: не были обнаружены сегменты для вычисления площади камня
                        if stone_area > 0:
                            if stone_area < 1:  # or not stone_already_present:
                                stone_area = 0

                            # каждое событие обновляется (добавляется площадь камня), а данные о нем отправляются на внешнее API
                            for event in event_list:
                                try:
                                    logger.debug(f'Event: {event.__dict__}')

                                    event = await Event.update(
                                        db_session=session,
                                        event_id=event.id,
                                        stone_area=str(stone_area)
                                    )

                                    if cfg.send_json:
                                        json = await Event.convert_event_to_json(
                                            db_session=session,
                                            event=event,
                                        )
                                        logger.debug(f'JSON:{json}')

                                        # отправка данных на внешний API
                                        await send_event_info(frame=frame, data=json,detection_time=detection_time)

                                except Exception as exc:
                                    logger.error(exc)
                                    break

                            # сбрасываем площадь камня
                            stone_area = 0
                            event_list.clear()

                logger.debug(f'results: {frame_pred}')

    # except StopIteration:
    except Exception as exc:
        logger.error(exc)

    return (saw_already_moving, stone_already_present, stone_history, stone_area_list, stone_area, event_list)


async def is_in_roi(
        roi_xyxy: List[tuple],
        object_xyxy: List[List[int]]
) -> bool:
    """
    Функция, позволяющая по координатам определить, находится ли
    обнраруженный объект в области интереса (частично или полностью).

    Args:
        roi_xyxy (List[tuple]): координаты области интереса в формате XYXY
        object_xyxy: List[List[int]]: координаты объекта на кадре в формате XYXY
    
    Returns:
        bool: возвращает True, если объект частично или полностью находится в области интереса, 
            и False, если коррдинаты никак не пересекаются
    """
    roi_xmin = int(roi_xyxy[0][0])
    roi_ymin = int(roi_xyxy[0][1])
    roi_xmax = int(roi_xyxy[1][0])
    roi_ymax = int(roi_xyxy[1][1])

    xmin, ymin, xmax, ymax = object_xyxy

    is_in_roi = False

    # проверяем, пересекаются ли координаты bbox'а с областью интереса
    if xmin <= roi_xmax and xmax >= roi_xmin and ymin <= roi_ymax and ymax >= roi_ymin:
        is_in_roi = True

    return is_in_roi


def calculate_motion(
        prev_bbox: List[float],
        curr_bbox: List[float]
) -> float:
    """
    Функция, позволяющая рассчитать величину смещения (евклидово расстояние)
    bbox'а относительно его местоположения на предыдущем кадре.

    Args:
        prev_bbox (List[float]): координаты bbox'а на текущем кадре в формате XYWH
        curr_bbox (List[float]): координаты bbox'а на предыдущем кадре в формате XYWH
    
    Returns:
        float: величина смещения bbox'а
    """
    prev_center = calculate_center(prev_bbox)
    curr_center = calculate_center(curr_bbox)

    motion = curr_center - prev_center
    magnitude = np.linalg.norm(motion)

    return magnitude


def calculate_center(
        bbox: List[float]
) -> np.ndarray:
    """
    Функция, позволяющая вычислить координаты центра bbox'а.

    Args:
        bbox (List[float]): координаты bbox'а в формате XYWH
    
    Returns:
        numpy.ndarray: координаты центра bbox'а
    """
    x, y, w, h = bbox
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    logger.debug(f'BBox center: {center_x}, {center_y}')

    return np.array([center_x, center_y])


def is_moving(
        magnitude: float,
        threshold: int
) -> bool:
    """
    Функция, позволяющая определить, движется ли объект
    на основании величины его смещения.

    Args:
        magnitude (float): величина смещения bbox'а
        threshold (int): предел, обозначающий величину, начиная с которой объект считается движущимся
    
    Returns:
        bool: возвращает True, если объект считается движущимся
    """
    in_motion = magnitude > threshold

    return in_motion


def calculate_segment_area(
        segment_coord: np.ndarray,
        ratio_px_to_cm: float = 1.2  # вычислено исходя из того, что диаметр пилы = 220 см
) -> float:
    """
    Функция, позволяющая рассчитать площадь сегмента в квадратных сантиметрах.

    Args:
        segment (List[float]/np.ndarray): координаты сегмента

        ratio_px_to_cm (float): коэффициент пересчета пикселей в сантиметры
            (по умолчанию 1.2, вычислено исходя из того,
             что диаметр пилы = 220 см в реальности и = 180 пикселей на кадре)
    
    Returns:
        float: площадь сегмента в квадратных сантиметрах
    """
    # contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_px = cv2.contourArea(segment_coord)
    logger.debug(f'Contour area: {area_px} px')

    area_cm2 = round(area_px / (ratio_px_to_cm ** 2))
    logger.debug(f'Slab area: {area_cm2} cm^2')

    return area_cm2


def convert_xywh_to_xyxy(
        bbox_xywh: np.ndarray
) -> list[float]:
    """
    Функция, позволяющая перевести координаты bbox'а из формата XYWH в XYXY.

    Args:
        bbox_xywh (np.ndarray): координаты bbox'а в формате XYWH
    
    Returns:
        np.ndarray: координаты bbox'а в формате XYXY
    """
    x, y, w, h = bbox_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


async def check_for_motion(
        db_session: AsyncSession,
        xywh_history: List[List[float]],
        detected_item: dict,
        saw_track_magn: float,
        already_moving: bool,
        curr_fps: int,
        detection_time: datetime,
        camera_id: int,
) -> tuple[float, bool, Event]:
    """
    Функция, позволяющая определить, движется ли объект (пила):
        - фиксируется величина смещения bbox'а за указанный
        в конфиг. файле отрезок времени;
        - если делается вывод о том, что объект находится в движении,
        однако до этого он был статичен, создается событие о начале движения объекта;
        - если делается вывод о том, что объект статичен, однако до этого он
        находился в движении, создается событие об остановке объекта.

    Args:
        db_session (AsyncSession): объект асинхронной сессии БД
        xywh_history (List[List[float]]): список, содержащий координаты bbox'а в формате XYWH на последовательных кадрах
        detected_item (dict): словарь, содержащий информацию об обнаруженном объекте, куда будет записано значение (bool), соответствующее движению объекта
        saw_track_magn (float): величина смещения bbox'а
        already_moving (bool): флаг, обозначающий, находился ли в движении объект до текущей проверки
        curr_fps (int): количество кадров, анализируемых за секунду видео
        detection_time (datetime): время обнаружения объекта
        camera_id (int): ID камеры
    
    Returns:
        кортеж, содержащий в себе:
            - saw_track_magn (float): обновленная величина смещения bbox'а
            - already_moving (bool): обновленное значение
            - event (Event): созданное событие
    """
    event = None

    xywh_history.append(detected_item['xywh'])

    if len(xywh_history) > 1:

        # собираем инфрмацию о движении объекта за заданный отрезок времени
        if len(xywh_history) < curr_fps * cfg.saw_moving_sec:
            magnitude = calculate_motion(
                prev_bbox=xywh_history[-2],
                curr_bbox=xywh_history[-1]
            )
            saw_track_magn += magnitude
        else:
            # если необходимое кол-во информации было собрано, делаем вывод о движении объекта
            logger.debug(f'Magnitude: {saw_track_magn}')
            in_motion = is_moving(
                magnitude=saw_track_magn,
                threshold=cfg.saw_moving_threshold
            )

            detected_item['saw_moving'] = in_motion
            logger.debug(f'Saw moving: {in_motion}')

            if already_moving == None:
                already_moving = in_motion
                logger.info(f'saw_already_moving set: {already_moving}')

            # если объект движется, но до этого был статичен, создаем событие о начале движения объекта
            elif in_motion and not already_moving:
                event = await Event.create(
                    db_session=db_session,
                    type_id=3,  # событие "Начало распила товарного блока". TODO rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'The saw started moving at {detection_time}, magnitude: {saw_track_magn}')
                already_moving = True

            # если объект статичен, но до этого был в движении, создаем событие об остановке объекта
            elif not in_motion and already_moving:
                event = await Event.create(
                    db_session=db_session,
                    type_id=4,  # событие "Окончание распила товарного блока". TODO rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'The saw stopped moving at {detection_time}, magnitude: {saw_track_magn}')
                already_moving = False

            # сбрасываем величину смещения bbox'а и историю координат bbox'а
            saw_track_magn = 0
            logger.debug('Saw magnitude nullified')
            xywh_history.clear()

    return saw_track_magn, already_moving, event


async def check_if_stone_present_or_transferred(
        db_session: AsyncSession,
        detected_class_ids: List[int],
        object_history: List[bool],
        object_already_present: bool,
        forklift_history: List[bool],
        saw_already_moving: bool,
        curr_fps: int,
        detection_time: datetime,
        camera_id: int,
        stone_id: int = 0,
        forklift_id: int = 2
) -> tuple[bool, list[bool], Event]:
    """
    Функция, позволяющая определить, присутствиует ли на видео объект (камень):
        - фиксируется наличие или отсутствие детекции объекта заданного класса 
        на протяжении указанного в конфиг. файле отрезка времени;
        - если делается вывод о том, что объект находится на видео,
        однако до этого его не было, создается событие о появлении нового объекта;
        - если делается вывод о том, что объекта на видео нет, однако до этого он
        присутствовал на видео, создается событие об отсутствии объекта.

    Args:
        db_session (AsyncSession): объект асинхронной сессии БД
        detected_class_ids (List[int]): список ID классов детекции, обнаруженных на кадре
        object_history (List[bool]): список значений, обозначающих наличие или отсутствие объекта на кадре
        object_already_present (bool): флаг, обозначающий, находился ли объект на видео до текущей проверки
        forklift_history (List[bool]): список значений, обозначающих наличие или отсутствие погрузчика на кадре
        saw_already_moving (bool): флаг, обозначающий, находится ли в движении пила
        curr_fps (int): количество кадров, анализируемых за секунду видео
        detection_time (datetime): время обнаружения объекта
        camera_id (int): ID камеры
        stone_id (int): ID класса камня
        forklift_id (int): ID класса погрузчика
    
    Returns:
        кортеж, содержащий в себе:
            - object_already_present (bool): обновленная величина смещения bbox'а
            - object_history (List[bool]): обновленная история наличия объекта
            - event (Event): созданное событие
    """
    event = None

    logger.debug(f'Class IDs: {detected_class_ids}')

    # определяем, был ли обнаружен на кадре камень, и обновляем историю значений
    if stone_id in detected_class_ids:
        object_history.append(True)
    else:
        object_history.append(False)

    if object_already_present == None:
        if len(object_history) >= curr_fps * 10:
            if object_history.count(True) > object_history.count(False):
                object_already_present = True
            else:
                object_already_present = False
            logger.info(f'object_already_present set: {object_already_present}')
    
    # определяем, был ли обнаружен на кадре погрузчик, и обновляем историю значений
    if forklift_id not in detected_class_ids:
        forklift_history.append(False)
    else:
        forklift_history.append(True)

        # если погрузчик был находится на видео на протяжении заданного отрезка времени,
        # проверяем, не был ли перемещен камень
        if len(forklift_history) >= curr_fps * cfg.forklift_present_threshold:

            # создаем событие о появлении нового камня, если были соблюдены все следующие условия:
            # - пила не находится в движении;
            # - погрузчик находится на видео на протяжении заданного отрезка времени (превышает заданный порог - допускаем погрешности детекции);
            # - флаг о наличии камня установлен на значении False;
            # - судя по истории, ранее камень не обнаруживался на видео на протяжении продолжительного отрезка времени,
            # - но камень стабильно обнаруживается на видео в последние секунды
            # ИЛИ
            # - флаг о наличии камня установлен на значении False, но камень стабильно обнаруживается в течение последней минуты.
            if (
                not saw_already_moving and
                forklift_history.count(True) > len(forklift_history) * cfg.majority_threshold and
                not object_already_present and
                object_history.count(False) > len(object_history) * cfg.majority_threshold and
                all(obj_present_result for obj_present_result in object_history[-(curr_fps * cfg.stone_change_threshold):])
            ) or (
                not object_already_present and
                all(obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
            ):

                event = await Event.create(
                    db_session=db_session,
                    type_id=1,  # событие "Новый товарный блок на станке". TODO rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'New stone detected at {detection_time}, event created: {event.__dict__}')
                object_already_present = True
            
            # создаем событие об отсутствии камня, если были соблюдены все следующие условия:
            # - пила не находится в движении;
            # - погрузчик находится на видео на протяжении заданного отрезка времени (превышает заданный порог - допускаем погрешности детекции);
            # - флаг о наличии камня установлен на значении True;
            # - судя по истории, ранее камень присутствовал на видео на протяжении продолжительного отрезка времени,
            # - но камень стабильно не обнаруживается на видео в последние секунды
            # ИЛИ
            # - флаг о наличии камня установлен на значении True, но камень стабильно отсутствует в течение последней минуты.
            elif (
                not saw_already_moving and
                forklift_history.count(True) > len(forklift_history) * cfg.majority_threshold and
                object_already_present and
                object_history.count(True) > len(object_history) * cfg.majority_threshold and
                all(not obj_present_result for obj_present_result in object_history[-(curr_fps * cfg.stone_change_threshold):])
            ) or (
                object_already_present and
                all(not obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
            ):

                event = await Event.create(
                    db_session=db_session,
                    type_id=2,  # событие "Товарный блок убран со станка". TODO rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'Stone removed at {detection_time}, event created: {event.__dict__}')
                object_already_present = False
    
    # region extra check if stone is present

    # дополнительная проверка на наличие камня, не зависищая от присутствия погрузчика в кадре:
    # создаем событие об обнаружении камня, если он стабильно обнаруживается в течение последней минуты
    if (
        object_already_present != None and
        not object_already_present and
        all(obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
    ):

        event = await Event.create(
            db_session=db_session,
            type_id=1,  # событие "Новый товарный блок на станке". TODO rm hardcoded id
            camera_id=camera_id,
            time=detection_time-timedelta(minutes=1)
        )
        logger.info(f'New stone detected at {detection_time-timedelta(minutes=1)}, event created: {event.__dict__}')
        object_already_present = True
    
    # создаем событие об отсутствии камня, если он стабильно не виден в течение последней минуты
    elif (
        object_already_present != None and
        object_already_present and
        all(not obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
    ):

        event = await Event.create(
            db_session=db_session,
            type_id=2,  # событие "Товарный блок убран со станка". TODO rm hardcoded id
            camera_id=camera_id,
            time=detection_time-timedelta(minutes=1)
        )
        logger.info(f'Stone removed at {detection_time-timedelta(minutes=1)}, event created: {event.__dict__}')
        object_already_present = False
    # endregion

    # очищаем списки (при превышении заданного порога)
    detected_class_ids.clear()

    if len(forklift_history) > curr_fps * cfg.forklift_history_threshold:
        forklift_history.clear()
    
    if len(object_history) > curr_fps * cfg.stone_history_threshold:
        object_history.clear()

    return object_already_present, object_history, event


async def get_stone_area(
    db_session: AsyncSession,
    frame: np.ndarray,
    seg_detector: Detector,
    stone_already_present: bool,
    # stone_history: List[bool],
    stone_area_list: List[float],
    stone_area: float,
    saw_already_moving: bool,
    curr_fps: int,
    camera_id: int,
    detection_time: datetime
) -> tuple[float, list[float]]:
    """
    Функция, вычисляющая площадт камня:
        - вычисляет площадь камня на протяжении заданного отрезка времени (на Х кадрах);
        - вычисляет среднее значение площади.

    Args:
        db_session (AsyncSession): объект асинхронной сессии БД
        frame (np.ndarray): кадр из видео
        seg_detector (Detector): модель, выполняющая сегментацию
        stone_already_present (bool): флаг, обозначающий, находился ли камень на видео до текущей проверки
        stone_history (List[bool]): список значений, обозначающих наличие или отсутствие камня на кадре
        stone_area_list (List[float]): список значений площади камня
        stone_area (float): площадь камня на кадре
        saw_already_moving (bool): флаг, обозначающий, находится ли в движении пила
        curr_fps (int): количество кадров, анализируемых за секунду видео
        camera_id (int): ID камеры
        detection_time (datetime): время обнаружения объекта
    
    Returns:
        кортеж, содержащий в себе:
            - stone_area (float): обновленная площадь камня в квадратных сантиметрах
            - stone_area_list (List[float]): обновленный список значений площади
    """
    # если каменя нет или он пропал, то обнуляем площадь и не вычисляем, пока не появится
    # if not stone_already_present:
    #     stone_area = 0
    #     stone_area_list.clear()

    # вычисляет площадь камня камня, если были соблюдены все следующие условия:
    # - площадь текущего камня не была вычислена ранее;
    # - флаг о наличии камня установлен на значении True;
    # - пила не находится в движении (для исключения оккулюзии);
    # - камень обнаружен в результатх детекции на текущем кадре;
    # - погрузчик отсутствует на кадре (для исключения оккулюзии);
    # - камень стабильно обнаруживается на видео в последние секунды.
    # if (
    #     stone_area == 0 and
    #     stone_already_present
    #     # not saw_already_moving
    #     # 0 in class_ids and  # stone class id
    #     # 2 not in class_ids and  # forklift class id
    #     # all(obj_present_result for obj_present_result in stone_history[-(curr_fps * 10):])
    # ):
    logger.debug(f'Stone list: {stone_area_list}')

    if len(stone_area_list) < cfg.max_stone_area_list:

        # пропускаем кадр через модель для сегментации
        seg_results = seg_detector.model(source=frame)

        # парсим результаты сегментации
        if seg_results and seg_results[0].masks:  # check if there are masks
            segment = seg_detector.parse_segmentation(seg_results)

            # проверяем, находится ли найденный сегмент в области интереса
            # item_in_roi = await is_in_roi(
            #                 roi_xyxy=camera_roi,
            #                 object_xyxy=item['xyxy']
            #             )
            # if item_in_roi:

            # отрисовываем сегментацию на кадре
            seg_detector.plot_segmentation(segment, frame, detection_time)

            # вычисляем площадь сегмента и сохраняем значение в список
            stone_area_prelim = calculate_segment_area(segment)
            if stone_area_prelim > 0:
                stone_area_list.append(stone_area_prelim)
        else:
            stone_area_list.append(0.1)  # check for no seg detection

    else:
        # если список содерижит достаточное кол-во значений, вычисляем среднее значение площади
        stone_area = round(np.average(stone_area_list), 1)
        logger.debug(f'Average stone area: {stone_area}')

        # обнуляем список значений площади
        stone_area_list.clear()
    
    return (stone_area, stone_area_list)


# region currently_not_used

async def find_class_objects_in_roi(
        roi_coord: List[tuple],
        class_id: int,
        result_dict: dict
) -> List[dict]:
    roi_xmin = int(roi_coord[0][0])
    roi_ymin = int(roi_coord[0][1])
    roi_xmax = int(roi_coord[1][0])
    roi_ymax = int(roi_coord[1][1])

    # is_in_roi = False
    prior_track_ids = []
    objects_in_roi = []

    for _, values in result_dict.items():
        # if values:
        for item in values:
            if item['class_id'] == class_id and item['track_id'] not in prior_track_ids:

                xyxy = item['xyxy']
                xmin, ymin, xmax, ymax = xyxy

                # check if the bounding box intersects or lies within the ROI
                if xmin <= roi_xmax and xmax >= roi_xmin and ymin <= roi_ymax and ymax >= roi_ymin:
                    logger.debug(f"Object with Class ID {class_id} is inside the ROI\n{item}")
                    prior_track_ids.append(item['track_id'])
                    objects_in_roi.append(item)
                else:
                    logger.debug(f"Object with Class ID {class_id} is outside the ROI\n{item}")

    return objects_in_roi


async def get_event_end_time(events: dict, track_id: int):

    # initialize with a value lower than the minimum time
    last_detection_time = datetime.strptime("01.01.1970 00:00:00", "%d.%m.%Y %H:%M:%S")
    for _, values in events.items():
        for item in values:
            if item["track_id"] == track_id:
                if item["time"] > last_detection_time:
                    last_detection_time = item["time"]

    return last_detection_time


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return iou

def detect_occlusion(detected_bbox, reference_bbox, size_threshold=0.7, iou_threshold=0.5):

    detected_bbox_xyxy = convert_xywh_to_xyxy(detected_bbox)
    reference_bbox_xyxy = convert_xywh_to_xyxy(reference_bbox)

    area_detected = detected_bbox[2] * detected_bbox[3]
    area_reference = reference_bbox[2] * reference_bbox[3]

    size_difference = area_detected / area_reference

    iou = calculate_iou(detected_bbox_xyxy, reference_bbox_xyxy)

    if size_difference < size_threshold and iou > iou_threshold:
        return True
    else:
        return False

# endregion
