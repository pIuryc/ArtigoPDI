import tensorflow as tf
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAÇÕES ---
MODEL_PATH = "models/xception_faces.h5"
TEST_DIR = "data/single/test"
IMG_SIZE = (299, 299)

# --- CARREGAR MODELO ---
print("🔹 Carregando modelo treinado...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- CARREGAR DADOS DE TESTE ---
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names
print(f"🔹 Classes detectadas: {class_names}")

# --- AVALIAÇÃO ---
print("\n🔹 Avaliando modelo no conjunto de teste...\n")
loss, acc = model.evaluate(test_ds)
print(f"✅ Acurácia no teste: {acc:.4f}")
print(f"✅ Perda (loss): {loss:.4f}")

# --- MATRIZ DE CONFUSÃO ---
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Matriz de Confusão - Teste")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

print("\n🔹 Relatório de classificação:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

