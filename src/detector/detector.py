import cv2
import csv
import numpy as np
import supervision as sv

from time import time
from datetime import timedelta
from collections import defaultdict
from ultralytics import YOLO
from clearml import Task
from sklearn.model_selection import train_test_split

import core.config as cfg
from core.logger import logger
# from tracker.tracker import Sort
from tracker.tracker_dpsort import Tracker
# from core.models import Event
from core.utils import extract_frame
from core.utils import get_time_from_video_path
from core.scenarios import find_class_objects_in_roi
from core.scenarios import get_event_end_time
from core.scenarios import calculate_motion
from core.scenarios import calculate_center
from core.scenarios import is_moving


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'mps'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
       
        # model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model = YOLO("best.pt")
        model.fuse()
    
        return model
    

    def train_custom(self, data):

        # train_X, val_X, train_y, val_y = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)

        task = Task.init(project_name="stone-cv", task_name="training02")

        results = self.model.train(
            data=data,
            epochs=10,
            batch=8,
            device='mps'
        )

        return results


    def predict_custom(self, frame):

        results = self.model(
            source=frame,
            device=self.device,
            conf=0.5
        )
        
        return results
    

    def predict_vid_showcase(self, video_path):

        results = self.model.predict(
        source=video_path,
        show=True,
        save=True,
        save_txt=True,
        device="mps",
        )
        
        return results
    

    def parse_detections(self, results):
        try:
            frame_pred = []
            for result in results:
                boxes = result.boxes.cpu().numpy()

                if boxes.cls.size > 0:  # and boxes.cls.any()
                    class_id = boxes.cls[0].astype(int)
                    conf = boxes.conf[0].astype(float)
                    xyxy = boxes.xyxy[0].tolist()
                    xywh = boxes.xywh[0].tolist()
                    track_id = boxes.id[0].astype(int)
                    logger.debug(f'class_id: {class_id} ({type(class_id)}), conf: {conf} ({type(conf)}), xyxy: {xyxy} ({type(xyxy)})')

                    prediction = {
                        "class_id": int(class_id),
                        "class_name": self.CLASS_NAMES_DICT[class_id],
                        # "track_id": 0,
                        "track_id": int(track_id),
                        "conf": round(float(conf), 2),  # invalid format for json
                        "time": 0,
                        "xyxy": xyxy,
                        "xywh": xywh
                    }
                    frame_pred.append(prediction)

                    detections = (np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])).reshape(-1, 5)

                else:
                    logger.debug('No detections')

                    detections = np.empty((0, 5))

                    # if class_id == 0.0:
                    #     xyxys.append(result.boxes.xyxy.cpu().numpy())
                    #     confidences.append(result.boxes.conf.cpu().numpy())
                    #     class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        except Exception as e:
            logger.error(e)
        
        return frame_pred, detections


    def save_detections_to_csv(self, results_dict, video_path, video_fps):
        try:
            # csv file
            file_path = f"{cfg.results_dir}/{video_path.split('/')[-1].split('.')[0]}.csv"

            with open(f'{file_path}', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Frame Index", "Class ID", "Class Name","Track ID", "Confidence", "Time", "XYXY", "XYWH"])

                for frame_index, frame_predictions in results_dict.items():

                    for prediction in frame_predictions:
                        class_id = prediction["class_id"]
                        class_name = prediction["class_name"]
                        track_id = prediction["track_id"]
                        conf = prediction["conf"]
                        xyxy = prediction["xyxy"]
                        xywh = prediction["xywh"]
                        detection_time = prediction["time"]

                        # if 'time' in prediction:
                        #     print(f'time: {prediction["time"]}')
                        #     detection_time = prediction["time"]
                        # else:
                        #     detection_time = 0

                        writer.writerow([frame_index, class_id, class_name, track_id, conf, detection_time, xyxy, xywh])

            logger.info(f'Detection results saved at {file_path}')

        except Exception as e:
            logger.error(e)
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
         # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            conf = boxes.conf[0]
            xyxy = boxes.xyxy[0]

            if class_id == 0.0:
              xyxys.append(result.boxes.xyxy.cpu().numpy())
              confidences.append(result.boxes.conf.cpu().numpy())
              class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    def __call__(self, video_path):

        frame_generator = extract_frame(
            video_path=video_path,
            fps=5
        )
        # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        # tracker = Tracker()
        vid_start_time, _ = get_time_from_video_path(video_path)
        all_results = {}
        track_history = defaultdict(lambda: [])
        
        try:
            for frame, frame_idx, video_fps in frame_generator:
                detection_time = vid_start_time + timedelta(seconds=frame_idx/video_fps)

                logger.debug(f'Frame ID: {frame_idx}')
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=0.5,
                    iou=0.5,
                    device='mps',
                    # tracker="bytetrack.yaml",
                    show=True
                )

                # results = self.predict_custom(
                #     frame=frame
                # )
                # detections = np.empty((0, 5))

                for result in results:
                    frame_pred, detections = self.parse_detections(result)

                    # add detection time
                    # if frame_pred:
                    for item in frame_pred:
                        item["time"] = detection_time
                        logger.debug(f'Detection time: {detection_time}')

                        # saw motion logic
                        if item['class_id'] == 1:
                            track_id = item["track_id"]
                            # x, y, w, h = item['xywh']
                            track = track_history[track_id]
                            track.append(item['xywh'])  # x, y center point

                            if len(track) > 1:
                                motion = calculate_motion(
                                    prev_bbox=track[-2],
                                    curr_bbox=track[-1]
                                )
                                in_motion = is_moving(motion=motion, threshold=1.5)

                                # if in_motion: create an event

                            # if len(track) > 30:  # retain 90 tracks for 90 frames
                            #     track.pop(0)

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

                    print(f'\n\n\n\nresults: {frame_pred}\n\n\n\n')
                    all_results[frame_idx] = frame_pred

        except StopIteration:
            pass

        objects_in_roi = []
        # for class_id in self.CLASS_NAMES_DICT.keys():
        objects = find_class_objects_in_roi(
            roi_coord=cfg.camera_1_roi,
            class_id=0,  # stone class id
            result_dict=all_results
        )
        objects_in_roi.extend(objects)

        for item in objects_in_roi:
            last_detection_time = get_event_end_time(all_results, item['track_id'])
            item['last_detection_time'] = last_detection_time
        logger.debug(f'Objects in ROI: {objects_in_roi}')

            # event = Event(
            #     type_id = 0,  # create event types
            #     start_time = item['time'],
            #     end_time = item['last_detection_time']
            # )

        self.save_detections_to_csv(
            results_dict=all_results,
            video_path=video_path,
            video_fps=video_fps
        )      

        # cap = cv2.VideoCapture(self.capture_index)
        # assert cap.isOpened()
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        # while True:
            # ret, frame = cap.read()
            # assert ret
        
            
            # results = self.predict(frame)
            # frame = self.plot_bboxes(results, frame)
            
            # end_time = time()
            # fps = 1/np.round(end_time - start_time, 2)
             
            # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            # cv2.imshow('YOLOv8 Detection', frame)
 
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
        
        # cap.release()
        # cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    detector = ObjectDetection(capture_index=0)  # add source
    detector()