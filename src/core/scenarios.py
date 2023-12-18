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

    last_detection_time = datetime.strptime("01.01.1970 00:00:00", "%d.%m.%Y %H:%M:%S")  # Initialize with a value lower than the minimum time
    for _, values in events.items():
        for item in values:
            if item["track_id"] == track_id:
                if item["time"] > last_detection_time:
                    last_detection_time = item["time"]

    return last_detection_time