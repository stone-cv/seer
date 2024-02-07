import cv2

from fastapi import FastAPI
from fastapi import APIRouter

from datetime import datetime
from datetime import timedelta

import core.config as cfg
from core.logger import logger
# from tracker.tracker import Sort
from detector.detector import ObjectDetection
from core.models import Camera
from core.database import SessionLocal
from core.utils import extract_frame
from core.utils import get_time_from_video_path
from core.scenarios import is_in_roi
from core.scenarios import check_for_motion
from core.scenarios import check_if_object_present


app = FastAPI(title="Seer")
api_router = APIRouter()

# @app.on_event('startup')
# async def app_startup() -> None:
#     """
#     Событие вызывается когда основное приложение было запущено

#     :return: None
#     """
#     pass


# @app.on_event('shutdown')
# async def app_shutdown() -> None:
#     """
#     Событие вызывается когда основное приложение было остановлено.

#     :return: None
#     """
#     pass

async def process_video_file(
    detector: ObjectDetection,
    video_path: str,
    camera_id: int
) -> None:
    
    frame_generator = extract_frame(
            video_path=video_path,
            fps=cfg.fps
        )

    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # tracker = Tracker()

    vid_start_time, _ = get_time_from_video_path(video_path)
    all_results = {}

    # variables for saw logic
    saw_xywh_history = []
    saw_track_magn = 0
    already_moving = False

    # variables for stone logic
    class_ids = []
    stone_history = []
    stone_already_present = False
    
    try:
        async with SessionLocal() as session:
            camera_roi = await Camera.get_roi_by_camera_id(
                db_session=session,
                camera_id=camera_id
            )

            for frame, frame_idx, video_fps, curr_fps in frame_generator:
                logger.debug(f'Frame ID: {frame_idx}')

                detection_time = vid_start_time + timedelta(seconds=frame_idx/video_fps)
                logger.debug(f'Detection time: {detection_time}')

                results = detector.track_custom(source=frame)
                # results = detector.predict_custom(source=frame)

                for result in results:
                    frame_pred, detections = detector.parse_detections(result)  # detections for an outside tracker

                    for item in frame_pred:
                        item['time'] = detection_time

                        # ROI check
                        item_in_roi = await is_in_roi(
                            roi_xyxy=camera_roi,
                            object_xyxy=item['xyxy']
                        )
                        if item_in_roi:
                            class_ids.append(item['class_id'])

                            # saw motion logic
                            if item['class_id'] == 1:  # saw class id
                                saw_track_magn, already_moving = await check_for_motion(
                                    db_session=session,
                                    xywh_history=saw_xywh_history,
                                    detected_item=item,
                                    saw_track_magn=saw_track_magn,
                                    already_moving=already_moving,
                                    curr_fps=curr_fps,
                                    detection_time=detection_time
                                )

                    # stone logic
                    stone_already_present = await check_if_object_present(
                        db_session=session,
                        class_id=0,  # stone class id
                        detected_class_ids=class_ids,
                        object_history=stone_history,
                        object_already_present=stone_already_present,
                        curr_fps=curr_fps,
                        detection_time=detection_time
                    )

                    # update tracker

                    # sort
                    # track_bbs_ids = tracker.update(detections)
                    # if track_bbs_ids.size != 0:
                        # track_id = int(track_bbs_ids[0][-1])
    
                    # deep_sort
                    # tracker.update(frame, detections)
                    # for track in tracker.tracks:
                    #     track_id = track.track_id
                    #     item["track_id"] = track_id
                    #     logger.debug(f'Track ID: {track_id}')

                logger.debug(f'results: {frame_pred}')
                all_results[frame_idx] = frame_pred

    # except StopIteration:
    except Exception as exc:
        logger.error(exc)

    detector.save_detections_to_csv(
        results_dict=all_results,
        video_path=video_path,
        video_fps=video_fps
    )


async def process_live_video(
    detector: ObjectDetection,
    camera_id: int
) -> None:

    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # tracker = Tracker()

    start_time = datetime.now()
    frame_idx = 1

    # variables for saw logic
    saw_xywh_history = []
    saw_track_magn = 0
    already_moving = False

    # variables for stone logic
    class_ids = []
    stone_history = []
    stone_already_present = False

    try:
        async with SessionLocal() as session:
            camera_roi = await Camera.get_roi_by_camera_id(  # just get camera object?
                db_session=session,
                camera_id=camera_id
            )
            camera_url = await Camera.get_url_by_camera_id(
                db_session=session,
                camera_id=camera_id
            )

            video_stream = cv2.VideoCapture(camera_url)
            video_fps = video_stream.get(cv2.CAP_PROP_FPS)

            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break

                detection_time = datetime.now()
                calculated_time = start_time + timedelta(seconds=frame_idx/video_fps)
                logger.debug(f'Detection time: {detection_time}, calculated time: {calculated_time}')

                results = detector.track_custom(source=frame)
                # results = detector.predict_custom(source=frame)

                for result in results:
                    frame_pred, detections = detector.parse_detections(result)  # detections for an outside tracker

                    for item in frame_pred:
                        item['time'] = detection_time

                        # ROI check
                        item_in_roi = await is_in_roi(
                            roi_xyxy=camera_roi,
                            object_xyxy=item['xyxy']
                        )
                        if item_in_roi:
                            class_ids.append(item['class_id'])

                            # saw motion logic
                            if item['class_id'] == 1:  # saw class id
                                saw_track_magn, already_moving = await check_for_motion(
                                    db_session=session,
                                    xywh_history=saw_xywh_history,
                                    detected_item=item,
                                    saw_track_magn=saw_track_magn,
                                    already_moving=already_moving,
                                    curr_fps=video_fps,
                                    detection_time=detection_time
                                )

                    # stone logic
                    stone_already_present = await check_if_object_present(
                        db_session=session,
                        class_id=0,  # stone class id
                        detected_class_ids=class_ids,
                        object_history=stone_history,
                        object_already_present=stone_already_present,
                        curr_fps=video_fps,
                        detection_time=detection_time
                    )

                    # update tracker

                    # sort
                    # track_bbs_ids = tracker.update(detections)
                    # if track_bbs_ids.size != 0:
                    # track_id = int(track_bbs_ids[0][-1])

                    # deep_sort
                    # tracker.update(frame, detections)
                    # for track in tracker.tracks:
                    #     track_id = track.track_id
                    #     item["track_id"] = track_id
                    #     logger.debug(f'Track ID: {track_id}')

                logger.debug(f'results: {frame_pred}')

                frame_idx += 1
                logger.debug(f'Frame index: {frame_idx}')

    except Exception as exc:
        logger.error(exc)
