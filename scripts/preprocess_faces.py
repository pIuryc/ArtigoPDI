import os
import json
import random
import shutil
import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

RAW_DIR = "data/raw"
OUT_FACES_DIR = "data/faces_raw"        # faces extraídas sem split
OUT_SPLIT_DIR = "data/faces_split"      # dataset final train/val/test

CLASS_NAMES = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx - Hairless Cat"]

CASCADE_PATH = "models/haarcascade_frontalcatface.xml"

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError("Baixe o Haarcascade: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml")

cascade = cv2.CascadeClassifier(CASCADE_PATH)

TARGET_SIZE = (299, 299)

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# limpeza de diretórios
for d in [OUT_FACES_DIR, OUT_SPLIT_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------
# FUNÇÃO: extrair faces com Haarcascade
# --------------------------------------------------------------

def extract_faces(img_path, save_dir, min_size=80):
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar:", img_path)
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size)
    )

    count = 0
    for i, (x, y, w, h) in enumerate(faces):
        pad = int(w * 0.25)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, TARGET_SIZE)

        out_path = os.path.join(save_dir, f"{os.path.basename(img_path)}_face{i}.jpg")
        cv2.imwrite(out_path, crop)
        count += 1

    return count


# --------------------------------------------------------------
# FUNÇÃO: data augmentation simples
# --------------------------------------------------------------

def augment_image(img):
    pil = Image.fromarray(img)

    # Lista de augmentações
    aug_imgs = []

    # flip horizontal
    aug_imgs.append(np.array(pil.transpose(Image.FLIP_LEFT_RIGHT)))

    # brilho +15%
    enhancer = Image.fromarray(img).convert("RGB")
    aug_imgs.append(np.clip(np.array(enhancer) * 1.15, 0, 255).astype("uint8"))

    # contraste leve
    aug_imgs.append(cv2.convertScaleAbs(img, alpha=1.2, beta=5))

    return aug_imgs


# --------------------------------------------------------------
# ETAPA 1 — extrair TODAS as faces
# --------------------------------------------------------------

print("\n================ EXTRATANDO FACES ================")

faces_per_class = {}

for cls in CLASS_NAMES:
    cls_raw_dir = os.path.join(RAW_DIR, cls)
    cls_faces_dir = os.path.join(OUT_FACES_DIR, cls)

    os.makedirs(cls_faces_dir, exist_ok=True)

    imgs = [f for f in os.listdir(cls_raw_dir) if f.lower().endswith((".jpg", ".png"))]

    total = 0

    for img_name in imgs:
        path = os.path.join(cls_raw_dir, img_name)
        n = extract_faces(path, cls_faces_dir)
        total += n

    faces_per_class[cls] = total
    print(f"{cls}: {total} faces extraídas")

# --------------------------------------------------------------
# ETAPA 2 — AUMENTAÇÃO DE DADOS
# --------------------------------------------------------------

print("\n================ AUMENTANDO O DATASET ================")

for cls in CLASS_NAMES:
    cls_dir = os.path.join(OUT_FACES_DIR, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]

    for img_name in imgs:
        path = os.path.join(cls_dir, img_name)
        img = cv2.imread(path)

        for j, aug in enumerate(augment_image(img)):
            out_path = os.path.join(cls_dir, f"{img_name}_aug{j}.jpg")
            cv2.imwrite(out_path, aug)

print("✔ Aumentação concluída!")


# --------------------------------------------------------------
# ETAPA 3 — DIVIDIR EM TRAIN / VAL / TEST
# --------------------------------------------------------------

print("\n================ DIVIDINDO EM TRAIN/VAL/TEST ================")

for folder in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(OUT_SPLIT_DIR, folder, cls), exist_ok=True)

for cls in CLASS_NAMES:
    cls_dir = os.path.join(OUT_FACES_DIR, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]

    random.shuffle(imgs)

    total = len(imgs)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    train_imgs = imgs[:train_end]
    val_imgs = imgs[train_end:val_end]
    test_imgs = imgs[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUT_SPLIT_DIR, "train", cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUT_SPLIT_DIR, "val", cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUT_SPLIT_DIR, "test", cls, img))

    print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")


print("\n=====================================================")
print("✅ Dataset FINAL criado em:", OUT_SPLIT_DIR)
print("=====================================================\n")
