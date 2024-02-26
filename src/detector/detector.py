import os
import cv2
import csv
import random
import shutil
import numpy as np
import supervision as sv

from time import time

from ultralytics import YOLO
from clearml import Task
from sklearn.model_selection import train_test_split

import core.config as cfg
from core.logger import logger


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
       
        # model = YOLO("yolov8l.pt")  # load a pretrained YOLOv8n model
        model = YOLO(cfg.weights_path)
        model.fuse()
    
        return model
    

    def split_dataset(self):
            dataset_txt_path = "datasets/obj_train_data"
            dataset_images_path = "datasets/images_train"
            train_path = "datasets/datasets/train"
            val_path = "datasets/datasets/val"

            train_ratio = 0.8

            annotation_files = os.listdir(dataset_txt_path)
            random.shuffle(annotation_files)

            num_files = len(annotation_files)
            num_train = int(num_files * train_ratio)

            train_files = annotation_files[:num_train]
            val_files = annotation_files[num_train:]

            for file in train_files:
                logger.debug(file)
                if file.startswith('vlcsnap') or file.startswith('cropped'):
                    shutil.move(os.path.join(dataset_txt_path, file), os.path.join(f'{train_path}/labels', file))
                    shutil.move(os.path.join(dataset_images_path, file.split(".")[0] + ".png"), os.path.join(f'{train_path}/images', file.split(".")[0] + ".png"))

            for file in val_files:
                logger.debug(file)
                if file.startswith('vlcsnap') or file.startswith('cropped'):
                    shutil.move(os.path.join(dataset_txt_path, file), os.path.join(f'{val_path}/labels', file))
                    shutil.move(os.path.join(dataset_images_path, file.split(".")[0] + ".png"), os.path.join(f'{val_path}/images', file.split(".")[0] + ".png"))


    def train_custom(self, data, split_required):

        task = Task.init(project_name="stone-cv", task_name=f"training_{time()}")

        if split_required:
            self.split_dataset()

        # results = self.model.train(
        #     data=data,
        #     epochs=10,
        #     batch=8,
        #     device='cpu'
        # )

        # return results


    def predict_custom(self, source):

        results = self.model(
            source=source,
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
    

    def track_custom(self, source):

        results = self.model.track(
            source=source,
            persist=True,
            conf=0.5,
            iou=0.5,
            device='mps',
            # tracker="bytetrack.yaml",
            show=True
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
                    logger.debug(f'class_id: {class_id}, track_id: {track_id}, conf: {conf}, xyxy: {xyxy}')

                    prediction = {
                        "class_id": int(class_id),
                        "class_name": self.CLASS_NAMES_DICT[class_id],
                        # "track_id": 0,
                        "track_id": int(track_id),
                        "conf": round(float(conf), 2),  # invalid format for json
                        "time": 0,
                        "xyxy": xyxy,
                        "xywh": xywh,
                        "saw_moving": None
                    }
                    frame_pred.append(prediction)
                    # detections = (np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])).reshape(-1, 5)  # for tracker

                else:
                    logger.info('No detections')
                    # detections = np.empty((0, 5))  # for tracker

        except Exception as e:
            logger.error(e)
        
        return frame_pred  # detections


    def save_detections_to_csv(self, results_dict, video_path, video_fps):
        try:
            # csv file
            file_path = f"{cfg.results_dir}/{video_path.split('/')[-1].split('.')[0]}.csv"

            with open(f'{file_path}', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Frame Index", "Class ID", "Class Name","Track ID", "Confidence", "Time", "XYXY", "XYWH", 'saw_moving'])

                for frame_index, frame_predictions in results_dict.items():

                    for prediction in frame_predictions:
                        class_id = prediction["class_id"]
                        class_name = prediction["class_name"]
                        track_id = prediction["track_id"]
                        conf = prediction["conf"]
                        xyxy = prediction["xyxy"]
                        xywh = prediction["xywh"]
                        detection_time = prediction["time"]
                        saw_moving = prediction["saw_moving"]

                        writer.writerow([frame_index, class_id, class_name, track_id, conf, detection_time, xyxy, xywh, saw_moving])

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
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            ret, frame = cap.read()
            assert ret
        
            start_time = time()
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    detector = ObjectDetection(capture_index=0)
    detector()
