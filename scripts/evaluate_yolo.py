import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INPUT_DIR = "data/multi_object"
LABELS_JSON = os.path.join(INPUT_DIR, "multi_object_labels.json")

detector = YOLO("models/yolov8n.pt")      
classifier = YOLO("models/yolo_cats.pt")  

CLASS_NAMES = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx - Hairless Cat"]

def expand_box(x1, y1, x2, y2, img_w, img_h, factor=0.30):
    w = x2 - x1
    h = y2 - y1

    x1e = max(0, int(x1 - w * factor))
    y1e = max(0, int(y1 - h * factor))
    x2e = min(img_w, int(x2 + w * factor))
    y2e = min(img_h, int(y2 + h * factor))

    return x1e, y1e, x2e, y2e


def classify_crop(crop):
    result = classifier.predict(
        crop,
        imgsz=320,
        conf=0.05,
        verbose=False
    )[0]

    if result.probs is None:
        return "Unknown"

    cls_id = int(result.probs.top1)
    return CLASS_NAMES[cls_id]


def detect_and_classify(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    detections = detector.predict(
        img,
        imgsz=1280,
        conf=0.05,
        iou=0.25,
        max_det=20,
        verbose=False
    )[0]

    preds = []

    if detections.boxes is None:
        return preds

    for box in detections.boxes:
        cls = int(box.cls[0])
        if cls != 15: 
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h)
        crop = img[ey1:ey2, ex1:ex2]

        pred = classify_crop(crop)
        preds.append(pred)

    return preds


def main():
    with open(LABELS_JSON, "r") as f:
        GT = json.load(f)

    all_true = []
    all_pred = []

    per_image_stats = {}  

    for filename, true_labels in GT.items():
        img_path = os.path.join(INPUT_DIR, filename)

        print(f"🔍 Avaliando {filename} (GT = {true_labels})")

        preds = detect_and_classify(img_path)
        print(f"→ Predições: {preds}")

        
        min_len = min(len(true_labels), len(preds))

        per_image_stats[filename] = {
            "gt": true_labels,
            "pred": preds,
            "correct": 0
        }

        for i in range(min_len):
            all_true.append(true_labels[i])
            all_pred.append(preds[i])

            if true_labels[i] == preds[i]:
                per_image_stats[filename]["correct"] += 1

   
    print(classification_report(all_true, all_pred, digits=3, zero_division=0))
    print(confusion_matrix(all_true, all_pred, labels=CLASS_NAMES))
    print("Acurácia total:", accuracy_score(all_true, all_pred))


    for k, v in per_image_stats.items():
        gt_count = len(v["gt"])
        correct = v["correct"]
        acc = correct / gt_count if gt_count > 0 else 0
        print(f"{k}: {correct}/{gt_count} = {acc:.2f}")


if __name__ == "__main__":
    main()
