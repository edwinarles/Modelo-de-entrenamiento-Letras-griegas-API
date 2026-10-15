# Clasificador de Letras Griegas Manuscritas (CNN en PyTorch)

Proyecto de Machine Learning para la clasificación de las 24 letras del alfabeto griego manuscritas utilizando una Red Neuronal Convolucional (CNN) en PyTorch entrenada con imágenes en escala de grises de 14x14 píxeles.

---

## 📁 Estructura del Proyecto

```
letras_uni-main/
├── download_dataset.py   # Descarga el dataset de Kaggle o genera uno sintético
├── train_model.py        # Arquitectura CNN, entrenamiento y validación
├── main.py               # Punto de entrada principal (evaluación, CLI y API REST sin HTML/CSS)
├── greek_letter_model.pth # Pesos del modelo entrenado (PyTorch)
├── model_metadata.json   # Metadatos del modelo (accuracy, clases, labels)
├── train.csv             # Dataset de entrenamiento (2400 muestras, 196 píxeles + label)
├── test.csv              # Dataset de prueba (480 muestras, 196 píxeles + label)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación
```

---

## 🔤 Clases y Mapeo de Letras (1 a 24)

| Label | Letra | Nombre | Label | Letra | Nombre | Label | Letra | Nombre |
|:---:|:---:|:---|:---:|:---:|:---|:---:|:---:|:---|
| 1 | α | alfa | 9 | ι | iota | 17 | ρ | rho |
| 2 | β | beta | 10 | κ | kappa | 18 | σ | sigma |
| 3 | γ | gamma | 11 | λ | lambda | 19 | τ | tau |
| 4 | δ | delta | 12 | μ | mu | 20 | υ | upsilon |
| 5 | ε | epsilon | 13 | ν | nu | 21 | φ | phi |
| 6 | ζ | zeta | 14 | ξ | xi | 22 | χ | chi |
| 7 | η | eta | 15 | ο | omicron | 23 | ψ | psi |
| 8 | θ | theta | 16 | π | pi | 24 | ω | omega |

---

## 🚀 Pasos para Ejecutar Todo el Código

### 1. Instalar Dependencias
Abre una terminal en la carpeta del proyecto y ejecuta:
```bash
pip install -r requirements.txt
```

### 2. (Opcional) Descargar o Regenerar el Dataset
Si deseas volver a generar la base de datos `train.csv` y `test.csv`:
```bash
python download_dataset.py
```
> Selecciona la **Opción 2** para generar el dataset sintético automáticamente con fuentes del sistema (sin necesidad de cuenta de Kaggle).

### 3. (Opcional) Reentrenar el Modelo
Para entrenar la CNN desde cero (50 epochs) y generar un nuevo `greek_letter_model.pth`:
```bash
python train_model.py
```
> El modelo actual alcanza un **98.1%** de accuracy en el conjunto de prueba (`test.csv`).

### 4. Probar la Clasificación en Consola
Para evaluar muestras del conjunto de prueba y verificar que las letras griegas se clasifiquen correctamente:
```bash
python main.py --test
```

Para clasificar un archivo de imagen específico:
```bash
python main.py --predict ruta_de_tu_imagen.png
```

### 5. Iniciar la API REST
Para ejecutar el servidor backend (sin interfaz HTML/CSS):
```bash
python main.py --serve
```
O simplemente:
```bash
python main.py
```
(El servidor iniciará en `http://localhost:5000`).

---

## 📡 Endpoints de la API REST

- **`GET /`**: Retorna el estado del servicio y la lista de endpoints disponibles en formato JSON.
- **`GET /model_info`**: Retorna información del modelo, precisión alcanzada y el mapeo de las 24 clases.
- **`GET /test_sample?label=1`**: Evalúa una muestra real de `test.csv` (opcionalmente filtrada por label 1 a 24) y retorna la predicción y el resultado de coincidencia.
- **`POST /predict`**: Clasifica una imagen. Acepta:
  - JSON con `"image"` (cadena base64).
  - JSON con `"pixels"` (arreglo de 196 valores numéricos).
  - Archivo multipart/form-data con la clave `"file"`.