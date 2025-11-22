# =============================================
# EXPORTAR BLOODMNIST A PNG Y DESCARGAR A LA PC
# =============================================

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO, BloodMNIST

script_dir = os.path.dirname(__file__)
# Carpeta donde se guardarán los PNG (relativa al script)
output_dir = os.path.join(script_dir, "bloodmnist_png")
os.makedirs(output_dir, exist_ok=True)

# Info de clases
info = INFO["bloodmnist"]
class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

def export_split(split_name):
    print(f"\nExportando split: {split_name}")

    ds = BloodMNIST(split=split_name, download=True)
    x = ds.imgs
    y = ds.labels.squeeze()

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Crear carpetas por clase
    for cname in class_names:
        os.makedirs(os.path.join(split_dir, cname), exist_ok=True)

    for i in range(len(x)):
        img = x[i]
        label = y[i]
        cname = class_names[label]

        filename = os.path.join(split_dir, cname, f"{split_name}_{i}.png")

        plt.imsave(filename, img)

        if i % 500 == 0:
            print(f"  Guardadas {i} imágenes...")

    print(f"✔ {split_name} exportado listo.")


# Exportar los 3 splits
export_split("train")
export_split("val")
export_split("test")

print("\n📦 Comprimiendo todo en un ZIP, espere...")

# Crear ZIP usando shutil (portable, funciona en Windows/Linux)
zip_base = os.path.join(script_dir, "bloodmnist_png")
zip_path = shutil.make_archive(zip_base, 'zip', root_dir=output_dir)
print(f"✔ ZIP creado: {zip_path}")

# En Windows podemos abrir el Explorador para la carpeta que contiene el ZIP
try:
    if sys.platform.startswith('win'):
        # Esto abrirá el archivo con la aplicación por defecto (explorer abrirá la carpeta)
        os.startfile(zip_path)
except Exception:
    pass

print("\n🎉 Listo. Archivo ZIP creado en:", zip_path)