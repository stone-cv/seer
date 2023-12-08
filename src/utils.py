import cv2


def extract_frame(video_path: str, fps: int = 5):
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)

    frame_interval = int(video_fps / fps)
    frame_idx = 0
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_count += 1
            yield frame, frame_idx
        frame_idx += 1

    video.release()
    print(f'Extracted {frame_count} frames from {video_path}')
    return
