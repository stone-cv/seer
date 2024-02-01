import numpy as np

from datetime import datetime
from typing import List

from core.logger import logger


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


# def check_for_motion(track_history, item, track_magn, already_moving, curr_fps, detection_time):
#     track_history.append(item['xywh'])

#     if len(track) > 1:
#         if len(track) < curr_fps*10:  # fps from frame_generator x 10 seconds
#             magnitude = calculate_motion(
#                 prev_bbox=track[-2],
#                 curr_bbox=track[-1]
#             )
#             track_magn += magnitude
#         else:
#             logger.debug(f'Magnitude: {track_magn}')
#             in_motion = is_moving(magnitude=track_magn, threshold=70)

#             item['saw_moving'] = in_motion
#             logger.debug(f'Saw moving: {in_motion}')

#             if in_motion and not already_moving:
#                 already_moving = True
#                 # create an event
#                 logger.info(f'The saw started moving at {detection_time}')

#             elif not in_motion and already_moving:
#                 already_moving = False
#                 # create an event
#                 logger.info(f'The saw stopped moving at {detection_time}')

#             # clear history
#             track_magn = 0
#             track.clear()
