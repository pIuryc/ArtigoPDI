import tensorflow as tf
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import os

# --- CONFIGURAÇÕES ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = "data/single/train"
VAL_DIR = "data/single/val"
MODEL_PATH = "models/xception_cats.h5"

# --- CARREGAR DATASETS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ⚠️ Guardar o número de classes antes do prefetch
num_classes = len(train_ds.class_names)
print(f"🔹 Número de classes detectadas: {num_classes} ({train_ds.class_names})")

# --- AUMENTO DE DADOS (AUGMENTATION) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

if os.path.exists(MODEL_PATH):
    print(f"🔹 Retomando treino a partir de {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    print("🔹 Nenhum modelo anterior encontrado. Iniciando do zero.")

# --- MODELO ---
base_model = Xception(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)  # ✅ corrigido
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar a base antes do fine-tuning
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS ---
os.makedirs("models", exist_ok=True)
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
]

# --- TREINAMENTO ---
print("\n🔹 Iniciando treinamento da cabeça da rede...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- FINE-TUNING ---
print("\n🔹 Fine-tuning (descongelando últimas camadas)...\n")
for layer in base_model.layers[-120:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# --- SALVAR MODELO FINAL ---
model.save(MODEL_PATH)
print(f"\n✅ Modelo salvo em: {MODEL_PATH}")
