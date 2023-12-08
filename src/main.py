import cv2
import csv
import json

from ultralytics import YOLO


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


def main():
    model = YOLO('yolov8n.pt')

    # # Train the model using the 'coco128.yaml' dataset for 3 epochs with 2 GPUs
    # results = model.train(data='coco128.yaml', epochs=1, imgsz=640, device='mps')

    # # Evaluate the model's performance on the validation set
    # results = model.val()

    frame_generator = extract_frame(video_path='static/baggage-on-belt.mov', fps=5)
    all_results = {}

    try:
        for frame, frame_idx in frame_generator:
            results = model(
                frame,
                # show=True,
                save=False,  # Disable saving images
                save_txt=False,  # Disable saving text files
                device='mps',
            )

            for result in results:
                frame_pred = []
                boxes = result.boxes.cpu().numpy()

                if boxes.cls.size > 0 and boxes.cls.any():
                    class_id = boxes.cls[0].astype(int)
                    conf = boxes.conf[0].astype(float)
                    xyxy = boxes.xyxy[0].tolist()
                    print(f'class_id: {class_id} ({type(class_id)}), conf: {conf} ({type(conf)}), xyxy: {xyxy} ({type(xyxy)})')

                    prediction = {
                        "class_id": int(class_id),
                        "conf": float(conf),  # invalid format for json
                        "xyxy": xyxy
                    }
                    frame_pred.append(prediction)
                    all_results[frame_idx] = frame_pred

                else:
                    print('No detections')

                # if class_id == 0.0:
                #     xyxys.append(result.boxes.xyxy.cpu().numpy())
                #     confidences.append(result.boxes.conf.cpu().numpy())
                #     class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

            # txt file with json dump
            # with open(f'results.txt', 'w') as f:
            #     json.dump(all_results, f)
            
            # csv file
            with open('results.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Frame Index", "Class ID", "Confidence", "XYXY"])

                for frame_index, frame_predictions in all_results.items():
                    for prediction in frame_predictions:
                        class_id = prediction["class_id"]
                        conf = prediction["conf"]
                        xyxy = prediction["xyxy"]
                        writer.writerow([frame_index, class_id, conf, xyxy])

    except StopIteration:
        pass

    # Export the model to ONNX format
    # success = model.export(
    #     format='onnx',
    #     device='mps'
    # )


if __name__ == '__main__':
    main()
