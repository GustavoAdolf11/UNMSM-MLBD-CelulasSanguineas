# proyectoCelulas

Resumen
-------
Entrena una CNN sobre BloodMNIST, muestra la arquitectura y genera una visualización LaTeX de la red usando PlotNeuralNet. Guarda el modelo en `model/`.

Requisitos
----------
- Windows 10/11
- Python 3.8+
- Git (en PATH)
- pdflatex (MiKTeX o TeX Live) si quieres generar PDF desde .tex
- Visual C++ Build Tools (si pip necesita compilar paquetes nativos)

Instalación (PowerShell)
------------------------
# proyectoCelulas

Resumen
-------
Este repositorio contiene scripts para entrenar y usar una CNN sobre el dataset BloodMNIST, visualizar la arquitectura con PlotNeuralNet y desplegar una pequeña interfaz con Gradio.

Objetivo
--------
Proveer instrucciones reproducibles y multiplataforma para:
- entrenar el modelo (`dominio.py`),
- generar una visualización LaTeX/PDF de la arquitectura (`PlotNeuralNet` + `dominio.py`),
- exportar el dataset a imágenes (`exportardatasetpng.py`),
- levantar una interfaz web para inspección de predicciones (`presentacionv2.py`).

Prerequisitos
-------------
- Python 3.8+ (recomendado 3.8–3.11)
- Git (opcional, necesario para clonar repositorios o instalar dependencias desde git)
- pdflatex (opcional, para generar PDF desde `.tex`; instala MiKTeX o TeX Live)
- En Windows, para compilar extensiones nativas puede necesitarse "Build Tools for Visual Studio" (Visual C++)

Instalación (genérica)
----------------------
1. Clona este repositorio y sitúate en la carpeta del proyecto:

```bash
git clone <https://github.com/GustavoAdolf11/UNMSM-MLBD-CelulasSanguineas.git>
cd <CelulasSanguineas>
```

2. Crea y activa un entorno virtual (elige la variante según tu OS):

# Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

# macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencias extra detectadas en el proyecto
-------------------------------------------
El proyecto usa (además de lo listado en `requirements.txt`) las siguientes dependencias que conviene tener en cuenta:

- gradio  (interfaz web)
- plotly  (gráficos interactivos en la GUI)

Si no están en `requirements.txt`, añádelas o instálalas manualmente:

```bash
pip install gradio plotly
```

PlotNeuralNet (visualización LaTeX)
-----------------------------------
PlotNeuralNet no siempre se instala correctamente vía pip porque el script espera la estructura de carpetas local (`pycore/tikzeng`). Tienes dos opciones:

- Clonar manualmente (recomendado):

```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

Coloca la carpeta `PlotNeuralNet/` en la raíz del proyecto para que `dominio.py` pueda importarla.

- Instalar desde git en `requirements.txt` (puede no dejar la carpeta física en la raíz):

```
PlotNeuralNet @ git+https://github.com/HarisIqbal88/PlotNeuralNet.git
# o en modo editable
-e git+https://github.com/HarisIqbal88/PlotNeuralNet.git#egg=PlotNeuralNet
```

Ejecución de scripts
---------------------
- Entrenar y generar visualización (entrenamiento completo):

```bash
python dominio.py
```

- Levantar la GUI (interfaz web con Gradio):

```bash
python presentacionv2.py
```

Esto abrirá una URL local en tu navegador. Si ejecutas en un servidor sin GUI, lanza Gradio con `server_name="0.0.0.0"` y un `server_port` explicito, o usa `share=True` para un túnel temporal.

- Exportar dataset a PNG y generar ZIP (sin Colab):

```bash
python exportardatasetpng.py
```

Salida y modelos
----------------
- El entrenamiento guarda el modelo en `model/bloodmnist_cnn.keras` y los nombres de clase en `model/bloodmnist_class_names.npy`.
- Estos archivos pueden ser grandes. Recomendaciones:
	- No añadir modelos pesados al repositorio si no es necesario.
	- Si quieres versionar modelos, usa Git LFS y un `.gitattributes` con reglas para `model/*.keras` y `model/*.npy`.

Smoke tests rápidos (no entrenan)
--------------------------------
Comprueba que el servicio y el modelo funcionan sin ejecutar todo el entrenamiento:

```bash
python - <<'PY'
from tensorflow.keras.models import load_model
from medmnist import BloodMNIST
import numpy as np

# carga modelo (si existe)
try:
		m = load_model('model/bloodmnist_cnn.keras')
except Exception as e:
		print('Modelo no disponible:', e)
		raise SystemExit(1)

# carga una imagen de prueba
ds = BloodMNIST(split='test', download=False)
x = ds.imgs[0].astype('float32')/255.0
pred = m.predict(x[np.newaxis, ...])
print('Predicción (clase):', np.argmax(pred))
PY
```

También puedes probar la función de servicio (si existe `model/servicio_predict.py`):

```bash
python - <<'PY'
from model.servicio_predict import predict_from_test
img, prob_dict, true_label, pred_label = predict_from_test(0)
print('True:', true_label, 'Pred:', pred_label)
PY
```

Reproducibilidad y versiones
----------------------------
- Para reproducibilidad, fija versiones en `requirements.txt`. Ejemplo de `requirements-freeze.txt`:

```bash
pip freeze > requirements-freeze.txt
```

- Si vas a publicar el repositorio con modelos binarios, utiliza Git LFS (instala `git-lfs` y configura `.gitattributes`):

```
model/*.keras filter=lfs diff=lfs merge=lfs -text
model/*.h5    filter=lfs diff=lfs merge=lfs -text
model/*.npy   filter=lfs diff=lfs merge=lfs -text
```

Problemas comunes y soluciones
-----------------------------
- ModuleNotFoundError: `servicio_predict` — verifica que `model/servicio_predict.py` existe y exporta `predict_from_test`.
- PlotNeuralNet no encontrado — clona `PlotNeuralNet/` en la raíz o instálalo desde git (ver sección arriba).
- pdflatex no encontrado — instala MiKTeX (Windows) o TeX Live (macOS/Linux) y añade `pdflatex` al PATH si quieres generar PDF desde `.tex`.
- Errores en `pip install` por compilación — instala las herramientas de compilación de tu sistema (Windows: Build Tools; Linux: build-essential).
- Si la GUI no aparece en el navegador: revisa la URL que imprime Gradio en la consola y verifica el firewall/puerto.

Estructura del proyecto (importante)
-----------------------------------
- `dominio.py` — entrenamiento y visualización de la arquitectura.
- `presentacionv2.py` — interfaz Gradio para inspeccionar predicciones.
- `servicio.py` / `servicio_predict.py` — funciones de servicio (carga de modelo, predicción).
- `exportardatasetpng.py` — exporta el dataset a PNG y crea un ZIP (local).
- `model/` — modelos y artefactos guardados.
- `PlotNeuralNet/` — (opcional) carpeta clonada para generar `.tex` via `pycore.tikzeng`.

Contribuir
----------
- Añade issues o pull requests con mejoras. Para cambios de código principales crea una rama por feature y abre PR contra `master`.
- Si añades modelos de ejemplo utiliza Git LFS o proporciona enlaces de descarga en lugar de subir binarios al repo.

Contacto
-------
Si quieres que actualice las instrucciones para un sistema concreto (Windows/macOS/Linux) o que fije versiones en `requirements.txt`, dímelo y lo hago.
