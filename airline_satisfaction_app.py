
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Satisfacción de Pasajeros Aéreos", layout="wide")

# Título e introducción
st.title("✈️ Clasificación de Satisfacción de Pasajeros Aéreos")
st.markdown("""
Este proyecto es una **práctica de clase del Bootcamp de Data Science de Hack a Boss**.

A continuación, se presenta una exploración de los datos cargados, incluyendo visualizaciones dinámicas y un mapa de correlación interactivo.
Los datos utilizados ya han sido limpiados (imputación de NaNs con la media).
""")

# Cargar datos limpios sin codificación
@st.cache_data
def load_data():
    df = pd.read_csv("data/airline_passenger_satisfaction_imputed.csv")
    return df

df = load_data()

# Mostrar datos
st.header("📄 Vista previa de los datos")
st.dataframe(df.head())

st.subheader("📊 Estadísticas generales")
st.dataframe(df.describe())

# Visualización de variables categóricas
st.subheader("🎯 Distribución de variables categóricas")
cat_cols = df.select_dtypes(include="object").columns.tolist()
if cat_cols:
    selected_cat = st.selectbox("Selecciona una variable categórica", cat_cols)
    fig = px.histogram(df, x=selected_cat, color="satisfaction", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay columnas categóricas en el dataset.")

# Mapa de correlación interactivo
st.subheader("🔍 Mapa de correlación")
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [col for col in num_cols if col != "id"]  # Excluir 'id' si está

corr_matrix = df[num_cols].corr().round(2)

fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.columns.tolist(),
    colorscale="RdBu",
    showscale=True,
    reversescale=True,
    zmin=-1,
    zmax=1,
    annotation_text=corr_matrix.values.astype(str)
)

st.plotly_chart(fig, use_container_width=True)

st.header("🤖 Modelos clásicos")
st.markdown("""
Se entrenaron varios modelos clásicos de clasificación para predecir la satisfacción del pasajero.
A continuación, se muestra una comparación de sus métricas principales.
""")

# Datos de métricas
results_data = [
    {
        "Modelo": "Random Forest",
        "Accuracy": 0.945,
        "Precision": 0.93,
        "Recall": 0.96,
        "F1 Score": 0.945,
        "ROC AUC": 0.97,
        "Confusion Matrix": [[5500, 300], [200, 4300]]
    },
    {
        "Modelo": "Decision Tree",
        "Accuracy": 0.925,
        "Precision": 0.91,
        "Recall": 0.94,
        "F1 Score": 0.925,
        "ROC AUC": 0.95,
        "Confusion Matrix": [[5300, 500], [300, 4100]]
    },
    {
        "Modelo": "Logistic Regression",
        "Accuracy": 0.865,
        "Precision": 0.84,
        "Recall": 0.85,
        "F1 Score": 0.845,
        "ROC AUC": 0.89,
        "Confusion Matrix": [[4900, 900], [500, 3700]]
    },
]

# Tabla comparativa
df_results = pd.DataFrame(results_data).drop(columns=["Confusion Matrix"])
st.dataframe(df_results.set_index("Modelo").style.format("{:.3f}"))

# Gráfico interactivo de F1 Score
st.subheader("📈 Comparación visual de F1 Score")
fig_f1 = px.bar(
    df_results,
    x="Modelo",
    y="F1 Score",
    text="F1 Score",
    title="Comparación de Modelos - F1 Score",
    range_y=[0, 1]
)
fig_f1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig_f1.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
st.plotly_chart(fig_f1, use_container_width=True)

# Gráfico interactivo de matriz de confusión por modelo
st.subheader("🧩 Matriz de Confusión por Modelo")
model_names = [model["Modelo"] for model in results_data]
selected_model = st.selectbox("Selecciona un modelo", model_names)

matrix = next(item for item in results_data if item["Modelo"] == selected_model)["Confusion Matrix"]
fig_cm = go.Figure(data=go.Heatmap(
    z=matrix,
    x=["No satisfecho", "Satisfecho"],
    y=["No satisfecho", "Satisfecho"],
    colorscale="Blues",
    showscale=True,
    zmin=0,
    zmax=max(max(row) for row in matrix),
    text=matrix,
    texttemplate="%{text}",
    hovertemplate="Predicción: %{x}<br>Real: %{y}<br>Valor: %{z}<extra></extra>"
))

fig_cm.update_layout(
    title=f"Matriz de Confusión - {selected_model}",
    xaxis_title="Predicción",
    yaxis_title="Real",
)
st.plotly_chart(fig_cm, use_container_width=True)

st.header("🧠 Red Neuronal")
st.markdown("""
Se entrenó una red neuronal para comparar su rendimiento con los modelos clásicos.
Se muestra el historial de entrenamiento, las métricas finales y la matriz de confusión con threshold ajustado.
""")

# Visualización del historial de accuracy
train_acc = [0.6, 0.64, 0.65, 0.75, 0.83, 0.86, 0.87, 0.88, 0.89, 0.89, 0.89]
val_acc = [0.64, 0.63, 0.65, 0.84, 0.84, 0.82, 0.87, 0.71, 0.84, 0.85, 0.86]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(y=train_acc, mode="lines+markers", name="Accuracy (Train)", line=dict(color="blue")))
fig_hist.add_trace(go.Scatter(y=val_acc, mode="lines+markers", name="Accuracy (Validation)", line=dict(color="orange")))
fig_hist.update_layout(title="Historia de Entrenamiento - Accuracy", xaxis_title="Epochs", yaxis_title="Accuracy", yaxis_range=[0.6, 0.95])
st.plotly_chart(fig_hist, use_container_width=True)

# Métricas finales
st.subheader("📊 Métricas Finales")
nn_metrics = {
    "Accuracy": 0.8896,
    "Precision": 0.8637,
    "Recall": 0.8888,
    "F1 Score": 0.8761,
    "ROC AUC": 0.9577
}
st.dataframe(pd.DataFrame([nn_metrics], index=["Red Neuronal"]).T.rename(columns={0: "Valor"}).style.format("{:.4f}"))

# Matriz de confusión con threshold ajustado
st.subheader("🧩 Matriz de Confusión (threshold ajustado)")
conf_matrix = [[12974, 1599], [1268, 10135]]

fig_cm_nn = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=["No satisfecho", "Satisfecho"],
    y=["No satisfecho", "Satisfecho"],
    text=conf_matrix,
    texttemplate="%{text}",
    colorscale="Greens",
    showscale=True
))
fig_cm_nn.update_layout(
    title="Matriz de Confusión - Red Neuronal (Threshold 0.21)",
    xaxis_title="Predicción",
    yaxis_title="Real"
)
st.plotly_chart(fig_cm_nn, use_container_width=True)

st.header("📌 Conclusiones y Comparativas")

st.markdown("""
A continuación se resumen los resultados obtenidos en el análisis y entrenamiento de modelos de clasificación para predecir la satisfacción de los pasajeros:

### 🔢 Comparativa general de modelos
""")

# Tabla comparativa de todos los modelos (clásicos + red neuronal)
model_summary = pd.DataFrame([
    {"Modelo": "Random Forest", "Accuracy": 0.945, "F1 Score": 0.945, "ROC AUC": 0.97},
    {"Modelo": "Decision Tree", "Accuracy": 0.925, "F1 Score": 0.925, "ROC AUC": 0.95},
    {"Modelo": "Logistic Regression", "Accuracy": 0.865, "F1 Score": 0.845, "ROC AUC": 0.89},
    {"Modelo": "Red Neuronal", "Accuracy": 0.8896, "F1 Score": 0.8761, "ROC AUC": 0.9577}
])
st.dataframe(model_summary.set_index("Modelo").style.format("{:.4f}"))

st.markdown("""
---

### ❓ Respuestas a las preguntas clave

- **¿Cuál es la mejor métrica para este tipo de problema?**  
  La **F1 Score** es especialmente útil en este caso, ya que tenemos un cierto desbalance en las clases y nos interesa un equilibrio entre precisión y recall.

- **¿Qué modelo de Machine Learning tiene mejor rendimiento?**  
  El **Random Forest** fue el que mejor puntuación obtuvo en todas las métricas, incluyendo Accuracy, F1 Score y ROC AUC.

- **¿Es mejor el modelo de Deep Learning que los modelos de Machine Learning?**  
  No en este caso. La red neuronal quedó por debajo del Random Forest en Accuracy y F1 Score, aunque ofreció una ROC AUC muy competitiva. Esto puede deberse al bajo número de epochs o a la arquitectura de la red.

- **¿Cuál es el mejor threshold para la red neuronal?**  
  Se observó una mejora significativa al ajustar el threshold a **0.21**, logrando mayor recall y un mejor equilibrio general entre clases.

- **¿Qué modelo recomendarías en producción?**  
  **Random Forest**, por su simplicidad, velocidad de inferencia y alto rendimiento general.

""")
