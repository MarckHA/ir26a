# Examen Primer Bimestre - Recuperación de Información (2026-A)

Este repositorio contiene la implementación de un Sistema de Recuperación de Información basado en similitud vectorial (embeddings) para el dataset *Rotten Tomatoes Movies and Critic Reviews*.

## Estructura del Repositorio
* `HernandezMark_ex1bim_ir26a.ipynb`: Jupyter Notebook principal con la ejecución completa del pipeline, evaluación de consultas y desafío de excelencia (PCA).
* `requirements.txt`: Archivo con las dependencias necesarias para ejecutar el sistema.

## Instrucciones de Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd <NOMBRE_DE_LA_CARPETA>

2. **Crear y activar un entorno virtual (Recomendado)**
   ```bash
   python -m venv .venv
    # En Windows:
    .venv\Scripts\activate
    # En macOS/Linux:
    source .venv/bin/activate

3. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt

4. **Ejecutar el Notebook**  
Inicia tu entorno, o abre VS Code.
Abre el archivo HernandezMark_ex1bim_ir26a.ipynb y ejecuta todas las celdas secuencialmente.


**Nota sobre los datos:** No es necesario descargar el corpus manualmente. El notebook utiliza la librería kagglehub para descargar la versión más reciente del dataset automáticamente durante la primera ejecución en la celda correspondiente.