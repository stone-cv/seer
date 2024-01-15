import cv2
from typing import Any
from datetime import datetime

from core.logger import logger


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
            yield frame, frame_idx, video_fps, fps
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



def get_camera_roi():  # doesn't work
    # Initialize variables
    drawing = False
    roi_points = []

    # Load the image or video frame
    frame = cv2.imread("static/0.jpg")

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

    # Display the frame and wait for the ROI to be defined
    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # Reset the ROI
            roi_points = []
            frame = cv2.imread("image.jpg")
        elif key == ord("c"):
            # Confirm the ROI and proceed with further processing
            break

    print(roi_points)
    cv2.destroyAllWindows()
    return roi_points


if __name__ == '__main__':
    get_camera_roi()
