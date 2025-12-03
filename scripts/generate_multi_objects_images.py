import os
import json
import random
from PIL import Image

RAW_DIR = "data/raw"            # onde estão as imagens individuais, separadas por pasta de classe
OUT_DIR = "data/multi_object"   # onde serão salvas as colagens
TARGET_SIZE = (299, 299)

os.makedirs(OUT_DIR, exist_ok=True)

# Nome das classes (mesmas do Xception e do YOLO)
CLASS_NAMES = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx - Hairless Cat"]


def load_all_images():
    """
    Carrega TODAS as imagens de data/raw agrupadas por classe.
    Estrutura esperada:
        data/raw/Bengal/*.jpg
        data/raw/Siamese/*.jpg
        ...
    """
    images_per_class = {}

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(RAW_DIR, class_name)
        imgs = []

        if not os.path.exists(class_dir):
            print(f"⚠️ Pasta ausente: {class_dir}")
            continue

        for f in os.listdir(class_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(class_dir, f))

        images_per_class[class_name] = imgs

    return images_per_class


def generate_collage(num_cats, images_per_class, out_id, labels_dict):
    """Gera a colagem + salva no JSON as raças reais."""
    
    # Escolhe aleatoriamente classes
    chosen_classes = random.choices(CLASS_NAMES, k=num_cats)

    # Seleciona uma imagem para cada classe escolhida
    selected_paths = []
    for cls in chosen_classes:
        img_path = random.choice(images_per_class[cls])
        selected_paths.append(img_path)

    # Carrega e redimensiona as imagens
    pil_images = []
    for p in selected_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize(TARGET_SIZE)
        pil_images.append(img)

    W = TARGET_SIZE[0] * num_cats
    H = TARGET_SIZE[1]

    collage = Image.new("RGB", (W, H), (255, 255, 255))

    x_offset = 0
    for img in pil_images:
        collage.paste(img, (x_offset, 0))
        x_offset += TARGET_SIZE[0]

    out_filename = f"{num_cats}cats_{out_id}.jpg"
    out_path = os.path.join(OUT_DIR, out_filename)
    collage.save(out_path)

    # ⚡ Salva o ground truth
    labels_dict[out_filename] = chosen_classes

    return out_path


def main():
    images_per_class = load_all_images()

    # JSON final
    labels = {}

    print("📸 Gerando colagens multi-objeto...")

    num_to_generate = 10  # por categoria

    for num_cats in [2, 3, 4, 5]:
        print(f"➡️ Gerando {num_to_generate} imagens com {num_cats} gatos...")
        for i in range(num_to_generate):
            generate_collage(num_cats, images_per_class, out_id=f"{num_cats}_{i}", labels_dict=labels)

    # Salva JSON com ground truth
    json_path = os.path.join(OUT_DIR, "multi_object_labels.json")
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=4)

    print("\n✅ Finalizado!")
    print("📁 Colagens salvas em:", OUT_DIR)
    print("📄 Ground truth salvo em:", json_path)


if __name__ == "__main__":
    main()
