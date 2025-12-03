import os
import cv2
import json
import itertools
from ultralytics import YOLO

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

detector = YOLO("yolov8n.pt")            # Detector COCO
classifier = YOLO("models/yolo_cats.pt") # Seu classificador de raças

CLASS_NAMES = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx - Hairless Cat"]

INPUT_DIR = "data/multi_object"
LABELS_JSON = ("data/multi_object/multi_object_labels.json")

OUT_DIR = "results_multi"
OUT_ANN = os.path.join(OUT_DIR, "annotated")
OUT_CROPS = os.path.join(OUT_DIR, "crops")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_ANN, exist_ok=True)
os.makedirs(OUT_CROPS, exist_ok=True)


# ==========================================================
# EXPANDIR BOX
# ==========================================================

def expand_box(x1, y1, x2, y2, img_w, img_h, factor=0.30):
    w = x2 - x1
    h = y2 - y1

    x1e = max(0, int(x1 - w * factor))
    y1e = max(0, int(y1 - h * factor))
    x2e = min(img_w, int(x2 + w * factor))
    y2e = min(img_h, int(y2 + h * factor))

    return x1e, y1e, x2e, y2e


# ==========================================================
# CLASSIFICAÇÃO YOLO-CATS
# ==========================================================

def classify_with_yolo_cats(crop):
    results = classifier.predict(crop, imgsz=320, conf=0.05, verbose=False)[0]

    if results.probs is None:
        return "Unknown", 0.0

    cls_id = int(results.probs.top1)
    conf = float(results.probs.top1conf)

    return CLASS_NAMES[cls_id], conf


# ==========================================================
# MATCHING ÓTIMO (SEM INDEXERROR)
# ==========================================================

def best_matching_score(preds, gts):
    np_ = len(preds)
    ng_ = len(gts)

    if np_ == 0:
        return 0

    k = min(np_, ng_)  # compara só até o mínimo

    best = 0
    for perm in itertools.permutations(range(ng_), k):
        score = sum(preds[i] == gts[perm[i]] for i in range(k))
        best = max(best, score)

    return best


# ==========================================================
# PROCESSAR UMA IMAGEM
# ==========================================================

def process_image(path):
    filename = os.path.basename(path)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    det = detector.predict(img, imgsz=1280, conf=0.05, iou=0.25, max_det=20, verbose=False)[0]

    preds = []
    ann = img.copy()
    count = 1

    if det.boxes is None:
        print("❌ Nenhum gato detectado:", filename)
        return []

    for box in det.boxes:
        cls = int(box.cls[0])
        if cls != 15:  # classe COCO 15 = cat
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # expandir box
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h)
        crop = img[ey1:ey2, ex1:ex2]

        if crop.size == 0:
            continue

        race, conf = classify_with_yolo_cats(crop)
        preds.append(race)

        # -------- SALVAR CROP --------
        crop_name = f"{filename}_cat{count}_{race}.jpg"
        cv2.imwrite(os.path.join(OUT_CROPS, crop_name), crop)

        # =============================
        # LABEL PEQUENA (AQUI É A MÁGICA)
        # =============================

        label = f"{race} ({conf:.2f})"

        # medir tamanho do texto
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        # calcular caixa pequena
        label_x1 = ex1
        label_y1 = ey1 - text_h - 10
        label_x2 = ex1 + text_w + 6
        label_y2 = ey1 - 4

        # impedir que vá para fora da imagem
        label_y1 = max(0, label_y1)

        # desenhar retângulo da label
        cv2.rectangle(ann, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)

        # desenhar texto em preto
        cv2.putText(
            ann, label,
            (label_x1 + 3, label_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )

        # bounding box do gato
        cv2.rectangle(ann, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)

        count += 1

    # salvar anotada
    cv2.imwrite(os.path.join(OUT_ANN, filename), ann)
    print(f"✓ Anotada em {os.path.join(OUT_ANN, filename)}")

    return preds


# ==========================================================
# MAIN + ACURÁCIA
# ==========================================================

def main():
    print("\n===============================================")
    print(" 🐱 YOLO DETECTOR + YOLO_CATS (ACURÁCIA FINAL) ")
    print("===============================================\n")

    # carregar ground-truth
    with open(LABELS_JSON, "r") as f:
        gt = json.load(f)

    results = {
        "2cats": {"correct": 0, "total": 0},
        "3cats": {"correct": 0, "total": 0},
        "4cats": {"correct": 0, "total": 0},
        "5cats": {"correct": 0, "total": 0},
    }

    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        if file not in gt:
            print(f"⚠️ Ignorado (não está no JSON): {file}")
            continue

        print(f"\n📌 Processando {file}")
        preds = process_image(os.path.join(INPUT_DIR, file))
        true_labels = gt[file]
        n = len(true_labels)

        group = f"{n}cats"
        results[group]["total"] += n

        if len(preds) == 0:
            print("⚠️ Nenhum gato detectado → 0 acertos")
            continue

        score = best_matching_score(preds, true_labels)
        results[group]["correct"] += score

        print(f"   → Detected={len(preds)} | GT={n} | Score={score}/{n}")

    print("\n\n===============================")
    print("         📊 RESULTADOS FINAIS ")
    print("===============================\n")

    for group, data in results.items():
        correct = data["correct"]
        total = data["total"]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{group}: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
