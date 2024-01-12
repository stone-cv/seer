import numpy as np

from datetime import datetime
from typing import List

from core.logger import logger


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

    return motion

def calculate_center(bbox):

    x, y, w, h = bbox
    center_x = x + (w / 2)
    center_y = y + (h / 2)

    logger.debug(f'Center: {center_x}, {center_y}')
    return np.array([center_x, center_y])

# Threshold for Movement
def is_moving(motion, threshold):

    magnitude = np.linalg.norm(motion)
    logger.debug(f'Saw magnitude: {magnitude}')
    in_motion = magnitude > threshold

    if in_motion:
        print("Saw is moving from side to side.")
    else:
        print("Saw is stationary.")

    return in_motion

# # Decision Making
# def make_decision(moving):

#     if moving:
#         print("Saw is moving from side to side.")
#     else:
#         print("Saw is stationary.")
