import os
import cv2
import numpy as np
import json
import time
from tqdm import tqdm
from ultralytics import YOLO

def extract_predictions(preds, image_id):
    detections = []
    for r in preds:
        for box in r.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            width = x_max - x_min
            height = y_max - y_min

            coco_bbox = [x_min, y_min, width, height]

            detection = {
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": coco_bbox,
                "score": score
            }
            detections.append(detection)
    return detections


def get_image_id(img_name):
    img_name = os.path.splitext(img_name)[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].replace('camera', ''))
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    return int(f"{cameraIndx}{sceneIndx}{frameIndx}")


def run_inference():
    engine_path = "/home/ktmt/Downloads/ICCV/best.onnx"
    test_dir = "/home/ktmt/Downloads/ICCV/images"
    output_json = "/home/ktmt/Downloads/ICCV/submission.json"

    model = YOLO(engine_path, task = "detect")

    images = sorted(os.listdir(test_dir))
    all_results = []

    for img_name in tqdm(images):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Resize ảnh về đúng kích thước input của model
        img = cv2.resize(img, (1280, 1280))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = model.predict(img, verbose=False)
        image_id = get_image_id(img_name)
        detections = extract_predictions(preds, image_id)
        all_results.extend(detections)

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)


if _name_ == "_main_":
    from eval_f1 import evaluate_f1

    start = time.time()
    run_inference()
    end = time.time()

    test_dir = "/home/ktmt/Downloads/ICCV/images"
    num_images = len([name for name in os.listdir(test_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fps = num_images / (end - start)
    max_fps = 25

    #f1 = evaluate_f1("submission.json", "ground_truth.json")
    norm_fps = min(fps, max_fps) / max_fps
    metric = 2 * norm_fps * f1 / (norm_fps + f1) if (norm_fps + f1) > 0 else 0

    print(f"FPS: {fps:.2f}")
    #print(f"F1-score: {f1:.4f}")
    #print(f"Final Metric: {metric:.4f}")