# ============================================
# 03_gui_gradio_v3.ipynb
# Capa de Presentación Profesional (V3)
# - Layout limpio y alineado
# - Imagen grande
# - Gráfico horizontal con % claros
# - Colores suaves tipo dashboard
# ============================================

import os, sys, random
BASE_DIR = os.path.join(os.path.dirname(__file__), 'model')
print("Carpeta de trabajo:", BASE_DIR)

# Importar la función de servicio
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from servicio_predict import predict_from_test

import gradio as gr
import plotly.graph_objects as go
import numpy as np


# ========= LÓGICA DE PRESENTACIÓN (wrapper) =========

def ui_predict(index: int):
    """
    Wrapper de la capa de servicio para la GUI.
    """
    img, prob_dict, true_label, pred_label = predict_from_test(index)

    # Ordenar clases por probabilidad (descendente)
    clases = list(prob_dict.keys())
    probs = [prob_dict[c] for c in clases]
    pares_ordenados = sorted(zip(clases, probs), key=lambda x: x[1], reverse=True)
    clases_ord, probs_ord = zip(*pares_ordenados)

    # Confianza máxima
    max_prob = probs_ord[0]
    max_clase = clases_ord[0]
    confianza_pct = max_prob * 100.0

    # Colores para las barras (predicción resaltada)
    colores = []
    for c in clases_ord:
        if c == pred_label:
            colores.append("#2563eb")  # azul principal
        else:
            colores.append("#94a3b8")  # gris azulado suave

    # Gráfico de barras horizontal con texto en %
    fig = go.Figure(
        data=[
            go.Bar(
                x=[p * 100 for p in probs_ord],
                y=list(clases_ord),
                orientation="h",
                marker_color=colores,
                text=[f"{p*100:.1f}%" for p in probs_ord],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Distribución de Probabilidades por Clase",
        xaxis_title="Probabilidad (%)",
        yaxis_title="",
        xaxis_range=[0, 105],
        template="simple_white",
        height=420,
        margin=dict(l=80, r=40, t=60, b=40),
    )

    # Estado: ¿acierto o error?
    if pred_label == true_label:
        estado = (
            "✅ **Predicción CORRECTA**\n\n"
            f"El modelo ha clasificado correctamente la célula como **{pred_label}**."
        )
    else:
        estado = (
            "⚠ **Predicción INCORRECTA**\n\n"
            f"Etiqueta real: **{true_label}**\n\n"
            f"Predicción del modelo: **{pred_label}**."
        )

    # Explicación amigable
    explicacion = (
        f"- La clase con mayor probabilidad es **{max_clase}**, "
        f"con una confianza de aproximadamente **{confianza_pct:.1f}%**.\n"
        f"- El gráfico horizontal muestra cómo se distribuye la probabilidad entre el resto de clases.\n"
        f"- Este panel permite revisar, caso por caso, el comportamiento del modelo sobre el conjunto de prueba."
    )

    return (
        img,                  # Imagen de la célula
        fig,                  # Gráfico de barras
        true_label,           # Etiqueta real
        pred_label,           # Predicción
        estado,               # Mensaje de estado
        explicacion,          # Explicación
        f"{confianza_pct:.1f} %"  # Confianza texto
    )


def indice_aleatorio():
    """Devuelve un índice aleatorio dentro del rango de test."""
    return random.randint(0, 3421 - 1)


# ========= INTERFAZ GRADIO (LAYOUT) =========

custom_css = """
body {
    background-color: #f3f4f6;
}
#panel-principal {
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}
.card {
    background-color: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    css=custom_css,
    title="BloodMNIST - Panel Profesional de Clasificación"
) as demo:

    with gr.Column(elem_id="panel-principal"):
        gr.Markdown(
            """
            # 🧬 Clasificador de Células Sanguíneas – BloodMNIST
            Panel profesional para analizar el comportamiento de la red neuronal convolucional
            entrenada sobre el dataset BloodMNIST.
            """
        )

        with gr.Row():
            # ===== Columna izquierda: control y métricas =====
            with gr.Column(scale=1, min_width=320):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 🔎 Selección de muestra")

                    index_input = gr.Slider(
                        0, 3421 - 1,
                        step=1,
                        value=0,
                        label="Índice de imagen en el conjunto de TEST"
                    )

                    with gr.Row():
                        boton_analizar = gr.Button("Analizar índice", variant="primary")
                        boton_random = gr.Button("Aleatorio")

                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📊 Métricas de predicción")

                    confianza_output = gr.Textbox(
                        label="Confianza de la predicción",
                        interactive=False
                    )

                    estado_output = gr.Markdown()

                    gr.Markdown(
                        """
                        **Uso sugerido**
                        - Explora distintos índices o usa el botón aleatorio.
                        - Observa la coincidencia entre etiqueta real y predicción.
                        - Revisa la distribución de probabilidades para interpretar la seguridad del modelo.
                        """
                    )

            # ===== Columna derecha: imagen + gráfico + detalle =====
            with gr.Column(scale=2, min_width=600):
                with gr.Group(elem_classes="card"):
                    with gr.Row():
                        img_output = gr.Image(
                            label="Imagen de la célula (TEST)",
                            type="numpy",
                            height=260
                        )

                    prob_plot_output = gr.Plot(
                        label="Probabilidades por clase"
                    )

                with gr.Group(elem_classes="card"):
                    with gr.Row():
                        true_output = gr.Textbox(
                            label="Etiqueta verdadera",
                            interactive=False
                        )
                        pred_output = gr.Textbox(
                            label="Predicción del modelo",
                            interactive=False
                        )

                    explicacion_output = gr.Markdown()

        # ===== Conexión de eventos =====

        # Clic en "Analizar índice"
        boton_analizar.click(
            fn=ui_predict,
            inputs=index_input,
            outputs=[
                img_output,
                prob_plot_output,
                true_output,
                pred_output,
                estado_output,
                explicacion_output,
                confianza_output
            ]
        )

        # Botón aleatorio => cambia el slider y dispara también el análisis
        boton_random.click(
            fn=indice_aleatorio,
            inputs=None,
            outputs=index_input
        ).then(
            fn=ui_predict,
            inputs=index_input,
            outputs=[
                img_output,
                prob_plot_output,
                true_output,
                pred_output,
                estado_output,
                explicacion_output,
                confianza_output
            ]
        )

demo.launch()