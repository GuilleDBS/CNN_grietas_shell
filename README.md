# Detección de Patologías Estructurales

Sistema de visión por computador basado en aprendizaje profundo (**Deep Learning**) para la detección, clasificación e interpretabilidad de patologías superficiales (fisuras y grietas) en estructuras de hormigón/concreto.

> [!NOTE]
> **Contexto Institucional**: Este proyecto fue desarrollado en el marco del **concurso Shell Lab en alianza con la Facultad de Ingeniería de la Universidad Nacional de Asunción (FIUNA), edición 2026**.

---

## 👥 Integrantes del Proyecto

- **Guillermo Benítez**
- **Micaela González**
- **Gustavo Añazco**
- **Marcio Martínez**
- **Axel López**

---

## ⚡ Recomendación de Entorno: Google Colab

> [!TIP]
> **Conviene ejecutar este proyecto en Google Colab**  
> Debido a que el dataset consta de **40.000 imágenes en alta resolución** y el modelo requiere entrenamiento convolucional intensivo, **se recomienda fuertemente ejecutar el notebook en [Google Colab](https://colab.research.google.com/)**:
> - **Aceleración por GPU gratuita**: Permite utilizar GPUs (como NVIDIA T4) reduciendo drásticamente el tiempo de entrenamiento de horas a pocos minutos.
> - **Descarga e integración directa**: Colab cuenta con un ancho de banda de alta velocidad para descargar y descomprimir el dataset directamente desde Kaggle sin saturar tu almacenamiento ni tu conexión local.
> - **Entorno preconfigurado**: Incluye TensorFlow, CUDA, cuDNN y librerías científicas listas para usar, evitando problemas de compatibilidad o dependencias en máquinas locales.
> - **Subida interactiva de `kaggle.json`**: El notebook implementa widgets nativos (`google.colab.files`) para cargar fácilmente las credenciales y probar imágenes externas.

---

## 📌 Descripción del Proyecto

El mantenimiento preventivo y la evaluación de la salud estructural en obras civiles (puentes, edificios, pavimentos, muros de contención) tradicionalmente dependen de inspecciones visuales manuales, las cuales son lentas, costosas y subjetivas.

Este proyecto implementa una solución automatizada mediante una **Red Neuronal Convolucional Personalizada (Custom CNN)** construida desde cero con **TensorFlow** y **Keras**, orientada a la clasificación binaria de superficies de hormigón:

- **`Positive`**: Superficie con presencia visible de fisuras o grietas estructurales.
- **`Negative`**: Superficie sana, sin grietas visibles.

### ✨ Características Principales

1. **Pipeline de Datos Robusto**:
   - Descarga y descompresión automatizada del conjunto de datos mediante la API de Kaggle.
   - Limpieza, estructuración en DataFrames y partición estratificada en conjuntos de **Entrenamiento (Train)**, **Validación (Val)** y **Prueba (Test)**.
   - Normalización y preprocesamiento con pipelines optimizados de `tf.data.Dataset` (prefetching y batching).
   - **Data Augmentation** integrado (rotaciones, traslaciones, volteos horizontal y vertical, ajustes de contraste) para evitar sobreajuste (*overfitting*).

2. **Arquitectura Custom CNN**:
   - Bloques convolucionales secuenciales con funciones de activación ReLU, padding reflectivo/same y reducción espacial mediante MaxPooling.
   - Capa de **GlobalAveragePooling2D** para reducir parámetros y mitigar sobreajuste.
   - Regularización con **Dropout** previo a la neurona final con activación Sigmoide.

3. **Evaluación Integral**:
   - Métricas detalladas: *Accuracy*, *Precision*, *Recall* y *F1-Score*.
   - Generación de matrices de confusión y gráficos de curvas de aprendizaje (Loss y Accuracy por época).

4. **Interpretabilidad con Grad-CAM (*Gradient-weighted Class Activation Mapping*)**:
   - Generación de mapas de calor visuales que resaltan qué regiones de la imagen influyeron en la decisión de la red, permitiendo validar que el modelo atiende a la fractura y no a artefactos del fondo.

5. **Inferencia con Imágenes Externas**:
   - Módulo interactivo para evaluar fotografías reales tomadas en campo fuera del conjunto original (como las muestras incluidas en el repositorio).

---

## 🔑 Configuración del archivo `kaggle.json` (Requisito Indispensable)

El notebook utiliza el dataset público [**Concrete Crack Images for Classification**](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification) de Arnav Sharma, el cual contiene **40.000 imágenes de 227×227 píxeles** (20.000 positivas y 20.000 negativas).

Para que la descarga automática funcione mediante el cliente oficial de Kaggle (`kaggle datasets download`), **es estrictamente necesario contar con el archivo de credenciales `kaggle.json`**.

### 1. ¿Cómo obtener el archivo `kaggle.json`?

1. Ingresa a [Kaggle](https://www.kaggle.com/) e inicia sesión con tu cuenta (si no tienes una, regístrate de forma gratuita).
2. Haz clic en tu foto de perfil (esquina superior derecha) y selecciona **Settings** (Configuración).
3. Desplázate hacia abajo hasta la sección **API**.
4. Haz clic en el botón **Create New Token** (o si ya existía uno previo, puedes regenerarlo con *Expire Old Token* y luego *Create New Token*).
5. Tu navegador descargará automáticamente un archivo llamado exactamente **`kaggle.json`**.

> [!WARNING]
> **Seguridad de tus credenciales:**  
> El archivo `kaggle.json` contiene tu nombre de usuario y tu clave API secreta (`username` y `key`). **Nunca subas este archivo a GitHub ni a ningún repositorio público**.

---

### 2. Uso en Google Colab (Recomendado)

Si ejecutas el cuaderno [`concrete-crack-detection-custom-cnn-colab-external-image.ipynb`](file:///c:/Users/Guille/Desktop/CNN_grietas_shell/concrete-crack-detection-custom-cnn-colab-external-image.ipynb) en Google Colab:

1. Abre el cuaderno en Colab y activa el acelerador por hardware (**Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**).
2. Ejecuta la celda de la sección **2. Descargar el dataset desde Kaggle**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
3. Se abrirá una ventana emergente en el navegador: selecciona y sube tu archivo local `kaggle.json`.
4. El script moverá el archivo a la ruta oculta `~/.kaggle/kaggle.json`, configurará los permisos de lectura (`chmod 600`) y ejecutará la descarga del dataset sin intervención adicional:
   ```bash
   kaggle datasets download -d arnavr10880/concrete-crack-images-for-classification -p /content/dataset --unzip
   ```

---

### 3. Uso en Entorno Local (Jupyter Notebook / Lab)

Si deseas ejecutar el proyecto localmente en tu ordenador:

- **En Windows:**
  Crea la carpeta `.kaggle` en tu directorio de usuario y copia allí el archivo:
  ```powershell
  mkdir "$HOME\.kaggle" -Force
  copy kaggle.json "$HOME\.kaggle\kaggle.json"
  ```
  Ruta resultante: `C:\Users\<TuUsuario>\.kaggle\kaggle.json`

- **En Linux / macOS:**
  ```bash
  mkdir -p ~/.kaggle
  cp kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```

---

## 📁 Estructura del Repositorio

```text
├── concrete-crack-detection-custom-cnn-colab-external-image.ipynb  # Notebook principal con el flujo completo
├── grieta_prueba.jpg                                               # Imagen externa de prueba 1
├── grieta_prueba2.jpg                                              # Imagen externa de prueba 2
├── grieta_prueba3.jpg                                              # Imagen externa de prueba 3
└── README.md                                                       # Documentación del proyecto
```

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Lenguaje**: Python 3.10+
- **Deep Learning / Visión**: TensorFlow 2.x, Keras
- **Procesamiento y Análisis de Datos**: NumPy, Pandas
- **Evaluación y Métricas**: Scikit-Learn
- **Visualización**: Matplotlib, Seaborn
- **Procesamiento de Imágenes**: Pillow (PIL), OpenCV
- **Automatización de Datos**: Kaggle API CLI

---

## 🚀 Flujo de Ejecución del Notebook

1. **Configuración inicial**: Carga de dependencias y verificación de disponibilidad de GPU CUDA.
2. **Autenticación y Descarga**: Carga de `kaggle.json` y extracción del dataset.
3. **Exploración**: Inspección balanceada de clases y visualización de muestras.
4. **Preprocesamiento**: Muestreo, división (Train / Val / Test) y pipeline de carga optimizado.
5. **Entrenamiento**: Ajuste de la Custom CNN con optimizador Adam, función de pérdida Binary Crossentropy y guardado de los mejores pesos.
6. **Evaluación de Rendimiento**: Matriz de confusión e informe de métricas de clasificación.
7. **Explicabilidad con Grad-CAM**: Generación de mapas de calor para verificar las regiones de activación de la red.
8. **Test con Imágenes Propias**: Carga de fotografías externas (como `grieta_prueba.jpg`) para predecir patologías y visualizar el Grad-CAM resultante.

---

> [!NOTE]
> **Aviso Técnico**: Las predicciones generadas por este modelo tienen fines de asistencia técnica, investigación y soporte preliminar. No sustituyen la inspección presencial ni el peritaje diagnóstico de un profesional matriculado en patología y cálculo estructural.
