from ultralytics import YOLO
import os
import shutil

# --- CONFIGURAÇÕES ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

EPOCHS_HEAD = 50      
EPOCHS_FINE = 30     

DATA_DIR = "data/single"     
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "yolo_cats.pt")

PROJECT = "runs_yolo"
RUN_NAME_HEAD = "cats_head"
RUN_NAME_FINE = "cats_finetune"


def get_classes_from_dir(train_dir):
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return classes

classes = get_classes_from_dir(TRAIN_DIR)
num_classes = len(classes)
print(f"🔹 Número de classes detectadas: {num_classes} ({classes})")

if os.path.exists(MODEL_PATH):
    print(f"🔹 Retomando treino a partir de {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print("🔹 Nenhum modelo anterior encontrado. Carregando YOLO pré-treinado (yolov8s-cls.pt).")
    model = YOLO("yolov8s-cls.pt")  # você pode trocar por yolov8n-cls.pt, etc.


print("\n🔹 Iniciando treinamento da cabeça da rede (camadas congeladas)...\n")

results_head = model.train(
    data=DATA_DIR,              
    imgsz=IMG_SIZE[0],          
    epochs=EPOCHS_HEAD,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=RUN_NAME_HEAD,
    patience=10,                
    lr0=1e-3,                  
    freeze=10,                  
    verbose=True
)

best_head_path = os.path.join(PROJECT, RUN_NAME_HEAD, "weights", "best.pt")
print(f"\n✅ Fim do treino da cabeça. Pesos salvos em: {best_head_path}")



print("\n🔹 Fine-tuning (descongelando camadas)...\n")

model_fine = YOLO(best_head_path)

results_fine = model_fine.train(
    data=DATA_DIR,
    imgsz=IMG_SIZE[0],
    epochs=EPOCHS_FINE,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=RUN_NAME_FINE,
    patience=10,
    lr0=5e-4,           
    freeze=0,           
    verbose=True
)

best_fine_path = os.path.join(PROJECT, RUN_NAME_FINE, "weights", "best.pt")
print(f"\n✅ Fine-tuning concluído. Melhor modelo em: {best_fine_path}")



if os.path.exists(best_fine_path):
    shutil.copy(best_fine_path, MODEL_PATH)
    print(f"\n✅ Modelo final copiado para: {MODEL_PATH}")
else:
    print("\n⚠️ Atenção: não encontrei o arquivo de pesos final para copiar!")
