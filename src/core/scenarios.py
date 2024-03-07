import os
import cv2
import numpy as np

from typing import List
from datetime import datetime
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

import core.config as cfg
from core.logger import logger
from tracker.tracker import Sort
from tracker.tracker_dpsort import Tracker
from detector.detector import ObjectDetection
from core.models import Event
from core.models import Camera
from core.database import SessionLocal
from core.utils import extract_frame
from core.utils import send_event_json
from core.utils import get_time_from_video_path


async def process_video_file(
    detector: ObjectDetection,
    video_path: str,
    camera_id: int,
    saw_already_moving: bool = None,
    stone_already_present: bool = None,
    stone_history: List[bool] = []
) -> tuple:
    """
    ???
    """
    tracker_sort = Sort(max_age=30, min_hits=7)  # iou_threshold=0.5  # sort
    tracker_dsort = Tracker()  # deep_sort

    vid_start_time, _ = get_time_from_video_path(video_path)

    # variables for saw logic
    saw_xywh_history = []
    saw_track_magn = 0

    # variables for stone logic
    class_ids = []
    forklift_history = []
    
    try:
        async with SessionLocal() as session:
            camera_roi = await Camera.get_roi_by_camera_id(
                db_session=session,
                camera_id=camera_id
            )

            frame_generator = extract_frame(
                video_path=video_path,
                camera_roi=camera_roi,
                fps=cfg.required_fps
            )

            for frame, frame_idx, video_fps, curr_fps in frame_generator:
                logger.debug(f'Frame ID: {frame_idx}')

                detection_time = vid_start_time + timedelta(seconds=frame_idx/video_fps)
                logger.debug(f'Detection time: {detection_time}')

                results = detector.predict_custom(source=frame)
                for result in results:
                    detections = detector.parse_detections(result)

                    # deep_sort
                    # tracker_dsort.update(frame, detections)
                    # for track in tracker.tracks:
                    #     logger.info(track.__dict__)
                    #     track_id = track.track_id
                    #     track_bbox = track.bbox
                    #     x1, y1, x2, y2 = track_bbox
                
                    # sort
                    tracked_obj = tracker_sort.update(detections)

                    for obj in tracked_obj:
                        track_id = int(obj[-2])
                        track_bbox = obj[:4]
                        class_id = int(obj[-1])
                        logger.debug(f'track_id: {track_id}, track_bbox: {track_bbox}, class_id: {class_id}')

                        # show frame
                        plotted_frame = detector.plot_bboxes(
                            frame=frame,
                            xyxy=track_bbox,
                            # conf=score,
                            class_id=class_id,
                            tracker_id=track_id,
                            roi_coords=camera_roi
                        )
                        cv2.imshow('Detection', plotted_frame)

                        # ROI check
                        item_in_roi = await is_in_roi(
                            roi_xyxy=camera_roi,
                            object_xyxy=track_bbox
                        )
                        if item_in_roi:
                            class_ids.append(class_id)

                            # saw motion logic
                            if class_id == 1:  # saw class id
                                saw_track_magn, saw_already_moving = await check_for_motion(
                                    db_session=session,
                                    xywh_history=saw_xywh_history,
                                    track_bbox=track_bbox,
                                    saw_track_magn=saw_track_magn,
                                    already_moving=saw_already_moving,
                                    curr_fps=curr_fps,
                                    detection_time=detection_time,
                                    camera_id=camera_id
                                )

                    # stone logic
                    stone_already_present, stone_history = await check_if_object_present(
                        db_session=session,
                        stone_id=0,  # stone class id
                        detected_class_ids=class_ids,
                        object_history=stone_history,
                        object_already_present=stone_already_present,
                        forklift_id=2,  # forklift class id
                        forklift_history=forklift_history,
                        curr_fps=curr_fps,
                        detection_time=detection_time,
                        camera_id=camera_id
                    )

    # except StopIteration:
    except Exception as exc:
        logger.error(exc)

    cv2.destroyAllWindows()

    return (saw_already_moving, stone_already_present, stone_history)


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

    # Check if the bounding box intersects or lies within the ROI
    if xmin <= roi_xmax and xmax >= roi_xmin and ymin <= roi_ymax and ymax >= roi_ymin:
        is_in_roi = True
        # logger.debug(f"The object detected is in the ROI")
    # else:
    #     logger.info(f"The object is detected not in the ROI")

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


def calculate_center(bbox: List[float]) -> np.ndarray:
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


def convert_xyxy_to_xywh(bbox_xyxy):
    x_min, y_min, x_max, y_max = bbox_xyxy
    width = x_max - x_min
    height = y_max - y_min
    x = x_min
    y = y_min
    return (x, y, width, height)


async def check_for_motion(
        db_session: AsyncSession,
        xywh_history: List[List[float]],
        # detected_item: dict,
        track_bbox: List[float],
        saw_track_magn: float,
        already_moving: bool,
        # last_motion: bool,  # TODO check for short motion
        curr_fps: int,
        detection_time: datetime,
        camera_id: int
) -> (float, bool):
    """
    Функция, позволяющая определить, движется ли объект:
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
    
    Returns:
        saw_track_magn (float): обновленная величина смещения bbox'а
        already_moving (bool): обновленное значение
    """
    bbox_xywh = convert_xyxy_to_xywh(track_bbox)
    xywh_history.append(bbox_xywh)

    if len(xywh_history) > 1:
        if len(xywh_history) < curr_fps * cfg.saw_moving_sec:
            magnitude = calculate_motion(
                prev_bbox=xywh_history[-2],
                curr_bbox=xywh_history[-1]
            )
            saw_track_magn += magnitude
        else:
            logger.debug(f'Magnitude: {saw_track_magn}')
            in_motion = is_moving(
                magnitude=saw_track_magn,
                threshold=cfg.saw_moving_threshold
            )

            # detected_item['saw_moving'] = in_motion
            logger.debug(f'Saw moving: {in_motion}')

            if already_moving == None:
                already_moving = in_motion
                logger.info(f'saw_already_moving set: {already_moving}')

            elif in_motion and not already_moving:
                # if not detect_occlusion(  # TODO verify against DEFINETLY not occluded bbox. reference size?
                #     detected_bbox=xywh_history[-2],
                #     reference_bbox=xywh_history[-1]
                # ):
                event = await Event.event_create(
                    db_session=db_session,
                    type_id=3,  # rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'The saw started moving at {detection_time}, magnitude: {saw_track_magn}')
                already_moving = True

                if cfg.send_json:
                    json = await Event.convert_event_to_json(
                        db_session=db_session,
                        event=event,
                    )
                    await send_event_json(data=json)

            elif not in_motion and already_moving:
                event = await Event.event_create(
                    db_session=db_session,
                    type_id=4,  # rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'The saw stopped moving at {detection_time}, magnitude: {saw_track_magn}')
                already_moving = False

                if cfg.send_json:
                    json = await Event.convert_event_to_json(
                        db_session=db_session,
                        event=event,
                    )
                    await send_event_json(data=json)

            # clear history
            saw_track_magn = 0
            logger.debug('Saw magnitude nullified')
            xywh_history.clear()

    return saw_track_magn, already_moving


async def check_if_object_present(
        db_session: AsyncSession,
        detected_class_ids: List[int],
        object_history: List[bool],
        object_already_present: bool,
        forklift_history: List[bool],
        curr_fps: int,
        detection_time: datetime,
        camera_id: int,
        stone_id: int = 0,
        forklift_id: int = 2
) -> bool:
    """
    Функция, позволяющая определить, присутствиует ли на видео объект:
        - фиксируется наличие или отсутствие детекции объекта заданного класса 
        на протяжении указанного в конфиг. файле отрезка времени;
        - если делается вывод о том, что объект находится на видео,
        однако до этого его не было, создается событие о появлении нового объекта;
        - если делается вывод о том, что объекта на видео нет, однако до этого он
        присутствовал на видео, создается событие об отсутствии объекта.

    Args:
        db_session (AsyncSession): объект асинхронной сессии БД
        stone_id (int): идентификатор класса детекции, к которому принадлежит объект
        detected_class_ids (List[int]): список идентификаторов классов детекции, обнаруженных на кадре
        object_history (List[bool]): список значений, обозначающих наличие или отсутствие обьъекта на кадре
        object_already_present (bool): флаг, обозначающий, находился ли объект на видео до текущей проверки
        curr_fps (int): количество кадров, анализируемых за секунду видео
        detection_time (datetime): время обнаружения объекта
    
    Returns:
        object_already_present (bool): обновленное значение
    """
    logger.debug(f'Class IDs: {detected_class_ids}')
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
    
    if forklift_id not in detected_class_ids:
        forklift_history.append(False)
    else:
        forklift_history.append(True)

        if len(forklift_history) >= curr_fps * cfg.forklift_present_threshold:

            if (
                forklift_history.count(True) > len(forklift_history) * cfg.majority_threshold and
                object_already_present != None and
                not object_already_present and
                object_history.count(False) > len(object_history) * cfg.majority_threshold and
                all(obj_present_result for obj_present_result in object_history[-(curr_fps * cfg.stone_change_threshold):])
            ) or (
                not object_already_present and
                all(obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
            ):

                event = await Event.event_create(
                    db_session=db_session,
                    type_id=1,  # rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'New stone detected at {detection_time}, event created: {event.__dict__}')
                object_already_present = True

                if cfg.send_json:
                    json = await Event.convert_event_to_json(
                        db_session=db_session,
                        event=event,
                    )
                    await send_event_json(data=json)

                # object_history.clear()
            
            elif (
                forklift_history.count(True) > len(forklift_history) * cfg.majority_threshold and
                object_already_present and
                object_history.count(True) > len(object_history) * cfg.majority_threshold and
                all(not obj_present_result for obj_present_result in object_history[-(curr_fps * cfg.stone_change_threshold):])
            ) or (
                object_already_present and
                all(not obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
            ):

                event = await Event.event_create(
                    db_session=db_session,
                    type_id=2,  # rm hardcoded id
                    camera_id=camera_id,
                    time=detection_time
                )
                logger.info(f'Stone removed at {detection_time}, event created: {event.__dict__}')
                object_already_present = False

                if cfg.send_json:
                    json = await Event.convert_event_to_json(
                        db_session=db_session,
                        event=event,
                    )
                    await send_event_json(data=json)

                # object_history.clear()
    
    # region extra check if stone is present
    if (
        object_already_present != None and
        not object_already_present and
        all(obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
    ):

        event = await Event.event_create(
            db_session=db_session,
            type_id=1,  # rm hardcoded id
            camera_id=camera_id,
            time=detection_time-timedelta(minutes=1)
        )
        logger.info(f'New stone detected at {detection_time-timedelta(minutes=1)}, event created: {event.__dict__}')
        object_already_present = True

        if cfg.send_json:
            json = await Event.convert_event_to_json(
                db_session=db_session,
                event=event,
            )
            await send_event_json(data=json)
    
    elif (
        object_already_present != None and
        object_already_present and
        all(not obj_present_result for obj_present_result in object_history[-(curr_fps * 60):])
    ):

        event = await Event.event_create(
            db_session=db_session,
            type_id=2,  # rm hardcoded id
            camera_id=camera_id,
            time=detection_time-timedelta(minutes=1)
        )
        logger.info(f'Stone removed at {detection_time-timedelta(minutes=1)}, event created: {event.__dict__}')
        object_already_present = False

        if cfg.send_json:
            json = await Event.convert_event_to_json(
                db_session=db_session,
                event=event,
            )
            await send_event_json(data=json)
    # endregion

    detected_class_ids.clear()

    if len(forklift_history) > curr_fps * cfg.forklift_history_threshold:
        forklift_history.clear()
    
    if len(object_history) > curr_fps * cfg.stone_history_threshold:
        object_history.clear()

    return object_already_present, object_history


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

                # Check if the bounding box intersects or lies within the ROI
                if xmin <= roi_xmax and xmax >= roi_xmin and ymin <= roi_ymax and ymax >= roi_ymin:
                    logger.debug(f"Object with Class ID {class_id} is inside the ROI\n{item}")
                    prior_track_ids.append(item['track_id'])
                    objects_in_roi.append(item)
                else:
                    logger.debug(f"Object with Class ID {class_id} is outside the ROI\n{item}")

    return objects_in_roi


async def get_event_end_time(events: dict, track_id: int):

    # Initialize with a value lower than the minimum time
    last_detection_time = datetime.strptime("01.01.1970 00:00:00", "%d.%m.%Y %H:%M:%S")
    for _, values in events.items():
        for item in values:
            if item["track_id"] == track_id:
                if item["time"] > last_detection_time:
                    last_detection_time = item["time"]

    return last_detection_time

# endregion
