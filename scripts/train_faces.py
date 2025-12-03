import tensorflow as tf
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# ==============================================================
# CONFIGURAÇÕES
# ==============================================================

IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 40              # treino da cabeça
FINE_TUNE_EPOCHS = 20    # fine-tuning

TRAIN_DIR = "data/faces_split/train"
VAL_DIR = "data/faces_split/val"
MODEL_PATH = "models/xception_faces.h5"

os.makedirs("models", exist_ok=True)

# ==============================================================
# CARREGAR DATASETS
# ==============================================================

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

num_classes = len(train_ds.class_names)
print(f"\n🔹 Classes detectadas ({num_classes}): {train_ds.class_names}")

# ==============================================================
# AUGMENTATION — otimizado para rostos
# ==============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==============================================================
# MODELO — criar do zero OU retomar
# ==============================================================

if os.path.exists(MODEL_PATH):
    print(f"\n🔹 Retomando treinamento do modelo salvo: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

else:
    print("\n🔹 Criando novo modelo Xception para faces...")

    base_model = Xception(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # congelar backbone
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

# ==============================================================
# CALLBACKS
# ==============================================================

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
]

# ==============================================================
# TREINO DA CABEÇA
# ==============================================================

print("\n===============================")
print("🔹 TREINANDO CAMADAS SUPERIORES")
print("===============================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================================================
# FINE-TUNING — destravar camadas finais da Xception
# ==============================================================

print("\n===============================")
print("🔹 FINE-TUNING (descongelando últimas camadas)")
print("===============================\n")

base_model = model.layers[1]  # camada Xception dentro do modelo final

# descongelar últimas 120 camadas (ajuste fino)
for layer in base_model.layers[-120:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# ==============================================================
# SALVAR MODELO FINAL
# ==============================================================

model.save(MODEL_PATH)
print(f"\n\n✅ MODELO FINAL SALVO EM: {MODEL_PATH}\n")
