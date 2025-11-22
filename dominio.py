# ============================================
# 01_train_model.ipynb
# Capa de Dominio: Entrenamiento del modelo CNN
# - Descarga BloodMNIST
# - Entrena CNN
# - Evalúa
# - Guarda el modelo entrenado en Google Drive
# ============================================

import os
BASE_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(BASE_DIR, exist_ok=True)
print("Carpeta de trabajo:", BASE_DIR)

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from medmnist import INFO, BloodMNIST
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import visualkeras
from IPython.display import display
from PIL import ImageFont
import sys
import subprocess
import shutil

print("Versión TensorFlow:", tf.__version__)

# Info del dataset
info = INFO["bloodmnist"]
print("\nDescripción:", info["description"])
print("Tarea:", info["task"])
print("Etiquetas:", info["label"])

# Cargar splits
train_ds = BloodMNIST(split="train", download=True)
val_ds   = BloodMNIST(split="val",   download=True)
test_ds  = BloodMNIST(split="test",  download=True)

# Pasar a NumPy y normalizar
x_train = train_ds.imgs.astype("float32") / 255.0
y_train = train_ds.labels.squeeze()

x_val   = val_ds.imgs.astype("float32") / 255.0
y_val   = val_ds.labels.squeeze()

x_test  = test_ds.imgs.astype("float32") / 255.0
y_test  = test_ds.labels.squeeze()

num_classes = len(np.unique(y_train))
input_shape = (28, 28, 3)

print("\nShapes:")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val  :", x_val.shape,   "y_val  :", y_val.shape)
print("x_test :", x_test.shape,  "y_test :", y_test.shape)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False,
    zoom_range=0.05
)

train_generator = datagen.flow(x_train, y_train, batch_size=64)

# Mostrar algunos ejemplos
plt.figure(figsize=(8, 3))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x_train[i])
    plt.title(f"Clase: {y_train[i]}")
    plt.axis("off")
plt.suptitle("Ejemplos de BloodMNIST")
plt.tight_layout()
plt.show()

# Definir la CNN (Dominio)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumen del modelo:")
model.summary()
# Visualización del modelo

vis = visualkeras.layered_view(
    model,
    options={'input_shape': input_shape},
    legend=True,
)
display(vis)

script_dir = os.path.dirname(__file__)
plotnn_dir = os.path.join(script_dir, "PlotNeuralNet")

if not os.path.isdir(plotnn_dir):
    raise FileNotFoundError(f"PlotNeuralNet no encontrado en {plotnn_dir}. Clónalo con: git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")

if plotnn_dir not in sys.path:
    sys.path.append(plotnn_dir)

from pycore.tikzeng import *

arch = [
    to_head('.'),
    to_cor(),
    to_begin(),

    # --- Conv 1 ---
    to_Conv("conv1", 32, 32, offset="(2,0,0)", to="(0,0,0)", height=32, depth=32, width=2, caption="Conv2D 32x3x3"),
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=28, depth=28, width=1, caption="MaxPool 2x2"),
    to_connection("conv1", "pool1"),

    # --- Conv 2 ---
    to_Conv("conv2", 64, 32, offset="(2,0,0)", to="(pool1-east)", height=24, depth=24, width=2, caption="Conv2D 64x3x3"),
    to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)", height=20, depth=20, width=1, caption="MaxPool 2x2"),
    to_connection("pool1", "conv2"),

    # --- Conv 3 ---
    to_Conv("conv3", 128, 32, offset="(2,0,0)", to="(pool2-east)", height=16, depth=16, width=2, caption="Conv2D 128x3x3"),
    to_Pool("pool3", offset="(1,0,0)", to="(conv3-east)", height=12, depth=12, width=1, caption="MaxPool 2x2"),
    to_connection("pool2", "conv3"),

    # --- BLOQUE DENSO + FLATTEN (representado de forma genérica) ---
    # Usamos to_SoftMax como bloque rectangular
    to_SoftMax("denseblock", 128, offset="(2,0,0)", to="(pool3-east)", caption="Flatten + Dense(128)"),
    to_connection("pool3", "denseblock"),

    # --- BLOQUE SOFTMAX FINAL ---
    to_SoftMax("softmax", 10, offset="(2,0,0)", to="(denseblock-east)", caption="Softmax"),
    to_connection("denseblock", "softmax"),

    to_end()
]

# Generar el .tex en el directorio del script
tex_path = os.path.join(script_dir, 'mi_red.tex')
to_generate(arch, tex_path)

# Ejecutar pdflatex en el directorio donde está el .tex (Windows: MiKTeX/TeX Live debe estar instalado y pdflatex en PATH)
pdflatex_exe = shutil.which("pdflatex")
if pdflatex_exe is None:
    print("Aviso: pdflatex no se encuentra en PATH. Instala MiKTeX o TeX Live para generar el PDF desde .tex.")
else:
    subprocess.run([pdflatex_exe, os.path.basename(tex_path)], cwd=script_dir, check=False)

cb = [
    callbacks.EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True, verbose=1)
]

# Entrenamiento
EPOCHS = 50  # puedes subir si quieres
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=2,
    callbacks=cb,
)

import matplotlib.pyplot as plt

# Curvas de precisión
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Curva de Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Curvas de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Curva de Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluación en TEST
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nResultados en TEST -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# Métricas detalladas
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nMATRIZ DE CONFUSIÓN:")
print(confusion_matrix(y_test, y_pred))

print("\nREPORTE DE CLASIFICACIÓN (precision, recall, f1 por clase):")
print(classification_report(y_test, y_pred, digits=4))

# Guardar modelo y nombres de clase en Google Drive
model_path = os.path.join(BASE_DIR, "bloodmnist_cnn.keras")
model.save(model_path)
print(f"\nModelo guardado en: {model_path}")

class_names = [info["label"][str(i)] for i in range(num_classes)]
class_names_path = os.path.join(BASE_DIR, "bloodmnist_class_names.npy")
np.save(class_names_path, np.array(class_names))
print(f"Nombres de clases guardados en: {class_names_path}")