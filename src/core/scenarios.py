import numpy as np

from typing import List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

import core.config as cfg
from core.logger import logger
from core.models import Event


def is_in_roi(roi_xyxy: List[tuple], object_xyxy: list) -> bool:
    roi_xmin = int(roi_xyxy[0][0])
    roi_ymin = int(roi_xyxy[0][1])
    roi_xmax = int(roi_xyxy[1][0])
    roi_ymax = int(roi_xyxy[1][1])

    xmin, ymin, xmax, ymax = object_xyxy

    is_in_roi = False

    # Check if the bounding box intersects or lies within the ROI
    if xmin <= roi_xmax and xmax >= roi_xmin and ymin <= roi_ymax and ymax >= roi_ymin:
        is_in_roi = True
        logger.debug(f"The object detected is in the ROI")
    else:
        logger.debug(f"The object is detected not in the ROI")

    return is_in_roi


def find_class_objects_in_roi(roi_coord: List[tuple], class_id: int, result_dict: dict):
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


def get_event_end_time(events: dict, track_id: int):

    # Initialize with a value lower than the minimum time
    last_detection_time = datetime.strptime("01.01.1970 00:00:00", "%d.%m.%Y %H:%M:%S")
    for _, values in events.items():
        for item in values:
            if item["track_id"] == track_id:
                if item["time"] > last_detection_time:
                    last_detection_time = item["time"]

    return last_detection_time


# Calculate Motion
def calculate_motion(prev_bbox, curr_bbox):

    prev_center = calculate_center(prev_bbox)
    curr_center = calculate_center(curr_bbox)

    motion = curr_center - prev_center
    magnitude = np.linalg.norm(motion)

    return magnitude

def calculate_center(bbox):

    x, y, w, h = bbox
    center_x = x + (w / 2)
    center_y = y + (h / 2)

    logger.debug(f'BBox center: {center_x}, {center_y}')
    return np.array([center_x, center_y])

# Threshold for Movement
def is_moving(magnitude, threshold):  # necessary?

    in_motion = magnitude > threshold

    return in_motion


async def check_for_motion(
        db_session: AsyncSession,
        xywh_history: list,
        detected_item: dict,
        saw_track_magn: float,  # ?
        already_moving: bool,
        curr_fps: int,
        detection_time: datetime
) -> None:  # ?

    xywh_history.append(detected_item['xywh'])

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

            detected_item['saw_moving'] = in_motion
            logger.debug(f'Saw moving: {in_motion}')

            if in_motion and not already_moving:
                event = await Event.event_create(
                    db_session=db_session,
                    type_id=3,  # rm hardcoded id
                    camera_id=1,  # default
                    time=detection_time
                )
                logger.info(f'The saw started moving at {detection_time}, event created: {event.__dict__}')
                already_moving = True

            elif not in_motion and already_moving:
                event = await Event.event_create(
                    db_session=db_session,
                    type_id=4,  # rm hardcoded id
                    camera_id=1,  # default
                    time=detection_time
                )
                logger.info(f'The saw stopped moving at {detection_time}, event created: {event.__dict__}')
                already_moving = False

            # clear history
            saw_track_magn = 0
            xywh_history.clear()

    return saw_track_magn, already_moving


async def check_if_object_present(
        db_session: AsyncSession,
        class_id: int,
        detected_class_ids: List[int],
        object_history: List[bool],
        object_already_present: bool,
        curr_fps: int,
        detection_time: datetime

) -> None:  # ?

    logger.debug(f'Class IDs: {detected_class_ids}')
    if class_id in detected_class_ids:
        object_history.append(True)
    else:
        object_history.append(False)
    detected_class_ids.clear()
    
    logger.debug(f'Stone history: {object_history}')
    if len(object_history) >= curr_fps * cfg.stone_check_sec:
        true_count = object_history.count(True)
        if true_count > len(object_history) // 2:
            stone_present = True
        else:
            stone_present = False
        
        if stone_present and not object_already_present:
            event = await Event.event_create(
                db_session=db_session,
                type_id=1,  # rm hardcoded id
                camera_id=1,  # default
                time=detection_time
            )
            logger.info(f'New stone detected at {detection_time}, event created: {event.__dict__}')
            object_already_present = True

        elif not stone_present and object_already_present:
            event = await Event.event_create(
                db_session=db_session,
                type_id=2,  # rm hardcoded id
                camera_id=1,  # default
                time=detection_time
            )
            logger.info(f'Stone removed at {detection_time}, event created: {event.__dict__}')
            object_already_present = False

        object_history.clear()

    return object_already_present
