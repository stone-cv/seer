import cv2
import csv
import numpy as np
import supervision as sv

from time import time
from ultralytics import YOLO

from logger import logger
from utils import extract_frame


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
       
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):

        try:
            results = self.model(source=frame, device=self.device)
        except StopIteration:
            pass
        
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
                    # id and is_track for tracking
                    logger.debug(f'class_id: {class_id} ({type(class_id)}), conf: {conf} ({type(conf)}), xyxy: {xyxy} ({type(xyxy)})')

                    prediction = {
                        "class_id": int(class_id),
                        "conf": float(conf),  # invalid format for json
                        "xyxy": xyxy
                    }
                    frame_pred.append(prediction)

                else:
                    logger.debug('No detections')

                    # if class_id == 0.0:
                    #     xyxys.append(result.boxes.xyxy.cpu().numpy())
                    #     confidences.append(result.boxes.conf.cpu().numpy())
                    #     class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        except Exception as e:
            logger.error(e)
        
        return frame_pred


    def save_detections_to_csv(self, results_dict):
        try:
            # txt file with json dump
            # with open(f'results.txt', 'w') as f:
            #     json.dump(all_results, f)
            
            # csv file
            with open('results/results.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Frame Index", "Class ID", "Confidence", "XYXY"])

                for frame_index, frame_predictions in results_dict.items():
                    for prediction in frame_predictions:
                        class_id = prediction["class_id"]
                        class_name = self.CLASS_NAMES_DICT[class_id]
                        conf = prediction["conf"]
                        xyxy = prediction["xyxy"]
                        writer.writerow([frame_index, class_id, class_name, conf, xyxy])

            logger.info('Detection results saved')

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
        start_time = time()
        all_results = {}
        
        try:
            for frame, frame_idx in frame_generator:
                logger.debug(f'Frame ID: {frame_idx}')
                results = self.model.predict(source=frame)
                frame_pred = self.parse_detections(results)
                all_results[frame_idx] = frame_pred
        except StopIteration:
            pass

        self.save_detections_to_csv(all_results)
                

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