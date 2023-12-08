from utils import extract_frame
from detector import ObjectDetection


def main():
    detector = ObjectDetection(capture_index=0)
    frame_generator = extract_frame('static/baggage-on-belt.mov')

    detector.save_detections(frame_generator)

if __name__ == '__main__':
    main()
