# ============================================
# 02_service_predict.ipynb
# Capa de Aplicación: crear módulo servicio_predict.py
# - Carga modelo y clases desde Google Drive
# - Carga dataset de TEST
# - Expone la función predict_from_test(index)
# ============================================

import os
BASE_DIR = os.path.join(os.path.dirname(__file__), 'model')
print("Carpeta de trabajo:", BASE_DIR)

# Verificar que el modelo y las clases existen
model_path = os.path.join(BASE_DIR, "bloodmnist_cnn.keras")
class_names_path = os.path.join(BASE_DIR, "bloodmnist_class_names.npy")

print("¿Existe modelo?:", os.path.exists(model_path))
print("¿Existe class_names?:", os.path.exists(class_names_path))

# Crear el archivo servicio_predict.py en Google Drive
# Este archivo define el servicio predict_from_test(index)

code = r'''
import os
import numpy as np
import tensorflow as tf
from medmnist import INFO, BloodMNIST

# Carpeta base en Google Drive
BASE_DIR = os.path.join(os.path.dirname(__file__), 'model')

# Cargar info y dataset de TEST
info = INFO["bloodmnist"]
test_ds = BloodMNIST(split="test", download=True)

x_test = test_ds.imgs.astype("float32") / 255.0
y_test = test_ds.labels.squeeze()

num_classes = len(np.unique(y_test))

# Cargar modelo entrenado y nombres de clases desde Drive
model_path = os.path.join(BASE_DIR, "bloodmnist_cnn.keras")
class_names_path = os.path.join(BASE_DIR, "bloodmnist_class_names.npy")

model = tf.keras.models.load_model(model_path)
class_names = np.load(class_names_path, allow_pickle=True).tolist()

def predict_from_test(index: int):
    """
    Servicio de aplicación.
    Recibe un índice del conjunto de test y devuelve:
    - imagen (uint8) para mostrar
    - diccionario de probabilidades por clase
    - etiqueta verdadera (texto)
    - predicción del modelo (texto)
    """
    # Asegurar rango
    index = int(index)
    index = max(0, min(index, len(x_test) - 1))

    # Imagen normalizada [0,1]
    img = x_test[index]

    # Para mostrarla (0–255 uint8)
    img_disp = (img * 255).astype("uint8")

    # Predicción
    img_batch = np.expand_dims(img, axis=0)
    probs = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    true_idx = int(y_test[index])
    pred_name = class_names[pred_idx]
    true_name = class_names[true_idx]

    prob_dict = {class_names[i]: float(probs[i]) for i in range(num_classes)}

    return img_disp, prob_dict, true_name, pred_name
'''

file_path = os.path.join(BASE_DIR, "servicio_predict.py")
with open(file_path, "w") as f:
    f.write(code)

print(f"Módulo de servicio creado en: {file_path}")

# 4) (Opcional) Probar el servicio importándolo desde aquí mismo
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from servicio_predict import predict_from_test

img, probas, true_label, pred_label = predict_from_test(0)
print("Prueba servicio:")
print(" - Etiqueta verdadera:", true_label)
print(" - Predicción modelo :", pred_label)