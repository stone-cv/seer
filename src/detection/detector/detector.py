import os
import cv2
import csv
import random
import shutil
import traceback
import numpy as np
import supervision as sv

import torch
from torchvision.transforms import v2
from PIL import Image

from time import time

from ultralytics import YOLO
from clearml import Task
from sklearn.model_selection import train_test_split

import core.config as cfg
from core.logger import logger


class Detector:

    def __init__(
            self,
            mode: str
    ) -> None:
        """
        Инициализация объекта детектора:
            - выбор устройства для работы (CUDA, если доступен GPU, иначе CPU),
            - загрузка нужной модели,
            - загрузка словаря с названиями классов,
            - создание словаря с идентификаторами классов (для упрощения дальнейшей работы).

        Args:
            mode (str): режим работы: 'det' для детекции или 'seg' для сегментации
        
        Returns:
            None
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'mps'  # 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model(mode)
        
        self.class_names_dict = self.model.model.names
        self.class_ids_dict = {val: key for key, val in self.class_names_dict.items()}
    
        # self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(
            self,
            mode: str
        ) -> YOLO:
        """
        Загрузка модели.
        В зависимости от указанноого режима работы выбираются нужные веса (путь прописан в конфиг. файле).

        Args:
            mode (str): 'det' для детекции или 'seg' для сегментации

        Returns:
            model (YOLO): модель
        """
       
        # model = YOLO("yolov8l.pt")  # l-models for init training
        if mode == 'det':
            model = YOLO(cfg.weights_det)
        if mode == 'seg':
            model = YOLO(cfg.weights_seg)

        model.fuse()
    
        return model
    
    def augment_dataset_dir(
        self,
        input_dir: str = 'src/datasets/obj_train_data',
        output_dir: str = 'src/datasets/seg/augmented'
    ) -> None:
        """
        Функция для аугментации изображений в датасете.
        Агументация производится с помощью torchvision.transforms:
        преобразовывается контраст, яркости и гамма (ч/б).

        Args:
            input_dir (str): путь до директории с исходными изображениями
            output_dir (str): путь до директории с аугментированными изображениями

        Returns:
            None
        """

        os.makedirs(output_dir, exist_ok=True)

        transform = v2.Compose([
            v2.ColorJitter(contrast=0.5, brightness=0.5),
            v2.RandomGrayscale(p=0.5)
        ])

        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert('RGB')

                augmented_img = transform(img)

                output_path = os.path.join(output_dir, filename)
                augmented_img.save(output_path)

                logger.debug(f'Processed {filename}')

        print('Data augmentation completed!')
    

    def split_dataset(
        self,
        dataset_txt_path: str = "src/datasets/obj_train_data",
        dataset_images_path: str = "src/datasets/images_train",
        train_path: str = "src/datasets/datasets/train",
        val_path: str = "src/datasets/datasets/val",
        train_ratio: float = 0.8
    ) -> None:
        """
        Функция для разделения датасета на обучающий и валидационный
        (по умолчанию в пропорции 80% для обучающей выборки и 20% для валидационной).

        Названия файлов в соответствии с конвенцией наименования скриншотов в VLC проигрывателе
        (+ обрезанные изображения), формат изображений - PNG.

        Args:
            dataset_txt_path (str): путь до директории с аннотациями
            dataset_images_path (str): путь до директории с изображениями
            train_path (str): путь до директории с обучающим датасетом
            val_path (str): путь до директории с валидационным датасетом
            train_ratio (float): пропорции обучающей выборки
        
        Returns:
            None
        """
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
        """
        Кастомная функция для обучения модели (с заданными параметрами).
        В целях логирования можно использовать таску в ClearML.
        При необходимости можно разделить датасет на обучающую и валидационную выборки.

        Args:
            data (str): путь до конфиг. файла
            split_required (bool): флаг разделения датасета на обучающую и валидационную выборки
        """
        # task = Task.init(project_name="stone-cv", task_name=f"training_{time()}")

        # разделение датасета на обучающую и валидационную выборки
        if split_required:
            self.split_dataset()

        results = self.model.train(
            data=data,
            epochs=50,
            batch=16,
            device=self.device,
            resume=True,
        )

        return results


    def predict_custom(self, source):
        """
        Кастомная функция для работы модели (предсказания) с заданными параметрами
        (низкая (0.3) уверенность обоснована низкой степенью ложноположитеных результатов детекции).

        Args:
            source (str): источник изображения (кадр)

        Returns:
            results: результаты детекции
        """
        results = self.model(
            source=source,
            device=self.device,
            conf=0.3,
            # show=True
        )
        
        return results
    

    def thread_safe_predict(self, mode, source):

        # should init a new model inside thread
        local_model = self.load_model(mode)

        results = local_model.predict(
            source=source,
            device=self.device,
            conf=0.3,
            # show=True
        )

        # implementation
        # results = Thread(
                #     target=Detector.thread_safe_predict,
                #     args=('det', frame, ),
                # ).start()

        return results
    

    def track_custom(self, source):
        """
        Кастомная функция для работы модели (предсказания) и трекера с заданными параметрами
        (низкая (0.3) уверенность обоснована низкой степенью ложноположитеных результатов детекции).

        Args:
            source: источник изображения (кадр)

        Returns:
            results: результаты детекции
        """
        results = self.model.track(
            source=source,
            persist=True,
            conf=0.3,
            iou=0.5,
            device=self.device,
            # tracker="bytetrack.yaml",
            show=True
        )

        return results
    

    def parse_detections(self, results) -> dict:
        """
        Парсинг результатов детекции.
        Из массива с результатами извлекаются:
            - ID класса,
            - коэффициент уверенности,
            - координаты bbox'а в формате XYXY (x1, y1, x2, y2),
            - координаты bbox'а в формате XYWH (x, y, ширина, высота),
            - ID объекта на основании трекера.
        По результатам парсинга формируется словарь с вышеперечисленными
        данными, а также такими параметрами, как время детекции объекта 
        и флаг движения объекта.

        Args:
            results: результаты детекции

        Returns:
            frame_pred (dict): словарь с распарсенными предсказаниями
        """
        try:
            frame_pred = []
            for result in results:
                boxes = result.boxes.cpu().numpy()

                if boxes.cls.size > 0:
                    class_id = boxes.cls[0].astype(int)
                    conf = boxes.conf[0].astype(float)
                    xyxy = boxes.xyxy[0].tolist()
                    xywh = boxes.xywh[0].tolist()

                    if boxes.id:
                        track_id = boxes.id[0].astype(int)
                    else:
                        track_id = 0
                    
                    logger.debug(f'class_id: {class_id}, track_id: {track_id}, conf: {conf}, xyxy: {xyxy}')

                    prediction = {
                        "class_id": int(class_id),
                        "class_name": self.class_names_dict[class_id],
                        # "track_id": 0,
                        "track_id": int(track_id),
                        "conf": round(float(conf), 2),
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
            logger.error(f'{e} {traceback.format_exc()}')
        
        return frame_pred
    

    def parse_segmentation(self, results):

        for result in results:
            # for mask, box in zip(result.masks.xy, result.boxes):
            for mask in result.masks.xy:
                mask_np = np.int32([mask])
                logger.debug(f'Mask coords: {type(mask_np)}')

            return mask_np
        
    
    def plot_segmentation(self, segment, image, detection_time):
        """
        Функция для отрисовки маски (в результате сегментации) на кадре.

        Args:
            segment: маска
            image: кадр
            detection_time: время детекции (для названия файла)

        Returns:
            plotted_img: изображение с отрисованной маской
        """
        # image_open = cv2.imread(image)

        # draw contour
        plotted_img = cv2.polylines(image, segment, True, (255, 0, 0), 1)

        # draw mask
        # mask_img = np.zeros_like(img)
        # cv2.drawContours(mask_img, segment, 0, (255, 0, 0), -1)
        # img = cv2.addWeighted(img_open, 1, mask_img, 0.5, 0)

        # cv2.fillPoly(img_open, segment, colors[color_number])  # crashes

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        filename = f"area_plotting/{detection_time.strftime('%m-%d-%Y_%H-%M-%S')}.png"
        cv2.imwrite(filename, plotted_img)

        return plotted_img


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
        self.labels = [f"{self.class_names_dict[class_id]} {confidence:0.2f}"
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
    detector = Detector()
    detector()
