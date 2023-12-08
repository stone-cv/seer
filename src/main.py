from utils import extract_frame
from detector import ObjectDetection

from logger import logger


def main():
    logger.info('App initiated')
    detector = ObjectDetection(capture_index=0)
    frame_generator = extract_frame('static/baggage-on-belt.mov')

    logger.info('Detection started')
    detector.save_detections(frame_generator)

if __name__ == '__main__':
    main()
