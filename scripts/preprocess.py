import os
import shutil
import random
import numpy as np
from PIL import Image
import tensorflow as tf

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/single"
TARGET_SIZE = (299, 299)

# Augmentação moderna
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

def clear_or_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

def augment_and_save(img_path, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE)

    img_tensor = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)

    aug_tensor = augmentation(img_tensor)[0].numpy().astype("uint8")
    aug_img = Image.fromarray(aug_tensor)

    aug_img.save(out_path)

def preprocess_dataset():
    print("🔹 Iniciando pré-processamento com augmentação + split...")

    # -------------------------
    # 1 — Coletar classes
    # -------------------------
    classes = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]

    if not classes:
        print("❌ Nenhuma classe encontrada em data/raw")
        return

    print("✅ Classes detectadas:", classes)

    # -------------------------
    # 2 — Criar estrutura
    # -------------------------
    clear_or_create_dir(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # -------------------------
    # 3 — Processar imagens
    # -------------------------
    total = 0

    for cls in classes:
        class_path = os.path.join(INPUT_DIR, cls)
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        total_imgs = len(images)
        train_limit = int(total_imgs * 0.8)
        val_limit = int(total_imgs * 0.9)  # 80% train + 10% val

        for i, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)

            if i < train_limit:
                split = "train"
            elif i < val_limit:
                split = "val"
            else:
                split = "test"

            out_path = os.path.join(OUTPUT_DIR, split, cls, img_name)

            augment_and_save(img_path, out_path)
            total += 1

    print(f"\n✅ Total de imagens processadas: {total}")
    print(f"📁 Saída final: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    preprocess_dataset()
