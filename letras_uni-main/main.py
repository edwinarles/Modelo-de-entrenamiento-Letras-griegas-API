"""
Clasificador de Letras Griegas Manuscritas.

Backend y punto de entrada principal (sin HTML/CSS).
Permite clasificar imágenes de letras griegas manuscritas usando un modelo CNN
entrenado en PyTorch con imágenes de 14x14 píxeles.

Uso por línea de comandos:
    python main.py                     # Ejecuta prueba de clasificación y lanza servidor API
    python main.py --test              # Prueba muestras de test.csv y muestra métricas
    python main.py --predict imagen.png # Clasifica un archivo de imagen específico
    python main.py --serve             # Inicia solo el servidor REST API
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO

# Configurar salida segura en consola para caracteres Unicode en Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

# Inicializar Flask
app = Flask(__name__)

# Configuración global
DEVICE = torch.device("cpu")
MODEL = None
MODEL_INFO = None

# Mapeo de labels (1-24) a letras griegas y nombres
LABEL_TO_LETTER = {
    1: 'α', 2: 'β', 3: 'γ', 4: 'δ', 5: 'ε', 6: 'ζ',
    7: 'η', 8: 'θ', 9: 'ι', 10: 'κ', 11: 'λ', 12: 'μ',
    13: 'ν', 14: 'ξ', 15: 'ο', 16: 'π', 17: 'ρ', 18: 'σ',
    19: 'τ', 20: 'υ', 21: 'φ', 22: 'χ', 23: 'ψ', 24: 'ω'
}

LETTER_NAMES = {
    1: 'alfa', 2: 'beta', 3: 'gamma', 4: 'delta', 5: 'epsilon', 6: 'zeta',
    7: 'eta', 8: 'theta', 9: 'iota', 10: 'kappa', 11: 'lambda', 12: 'mu',
    13: 'nu', 14: 'xi', 15: 'omicron', 16: 'pi', 17: 'rho', 18: 'sigma',
    19: 'tau', 20: 'upsilon', 21: 'phi', 22: 'chi', 23: 'psi', 24: 'omega'
}


def load_model():
    """Carga el modelo CNN entrenado."""
    global MODEL, MODEL_INFO

    from train_model import GreekLetterCNN

    model_path = os.path.join(os.path.dirname(__file__), 'greek_letter_model.pth')
    if not os.path.exists(model_path):
        print(f"ADVERTENCIA: No se encontro el archivo del modelo en {model_path}")
        print("Ejecuta primero: python train_model.py")
        return False

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        num_classes = checkpoint.get('num_classes', 24)
        MODEL = GreekLetterCNN(num_classes).to(DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()

        MODEL_INFO = {
            'num_classes': num_classes,
            'accuracy': float(checkpoint.get('test_accuracy', 0.0)),
            'epoch': int(checkpoint.get('epoch', 0)),
            'model_path': model_path,
        }

        print(f"Modelo cargado exitosamente (Accuracy en test: {MODEL_INFO['accuracy']:.1%})")
        return True
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return False


def preprocess_image(image_input):
    """
    Preprocesa una entrada de imagen a un tensor de PyTorch (1, 1, 14, 14).
    Acepta:
      - numpy.ndarray o lista de 196 píxeles (formato CSV)
      - numpy.ndarray de 2D/3D
      - PIL.Image
      - Cadena en base64 (con o sin encabezado data:image/png;base64,)
      - Ruta de archivo en el sistema
    """
    raw_img = None

    if isinstance(image_input, (list, np.ndarray)):
        arr = np.array(image_input, dtype=np.float32)
        if arr.size == 196:
            # Formato de 196 características de 14x14
            arr = arr.reshape(14, 14)
            # Si tiene fondo blanco (típico en CSV ~255), invertir a trazo blanco / fondo negro
            corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
            if np.mean(corners) > 127:
                arr = 1.0 - (arr / 255.0)
            elif arr.max() > 1.0:
                arr = arr / 255.0
            return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
        elif arr.ndim in (2, 3):
            raw_img = Image.fromarray(arr.astype(np.uint8))
        else:
            raise ValueError(f"Tamano de arreglo invalido: {arr.shape}")

    elif isinstance(image_input, str):
        if image_input.startswith("data:image") or (len(image_input) > 250 and not os.path.exists(image_input)):
            # Cadena base64
            cleaned_b64 = image_input.split(",")[-1]
            raw_img = Image.open(BytesIO(base64.b64decode(cleaned_b64)))
        elif os.path.exists(image_input):
            # Archivo en disco
            raw_img = Image.open(image_input)
        else:
            raise FileNotFoundError(f"No se encontro el archivo o formato base64 invalido: {image_input[:50]}")

    elif isinstance(image_input, Image.Image):
        raw_img = image_input

    elif hasattr(image_input, 'read'):
        # Archivo en memoria (e.g., Flask request.files)
        raw_img = Image.open(image_input)

    else:
        raise TypeError(f"Tipo de entrada no soportado: {type(image_input)}")

    # Convertir a escala de grises manejando canal alfa
    if raw_img.mode in ('RGBA', 'LA') or (raw_img.mode == 'P' and 'transparency' in raw_img.info):
        raw_img = raw_img.convert('RGBA')
        bg = Image.new('RGB', raw_img.size, (255, 255, 255))
        bg.paste(raw_img, mask=raw_img.split()[3])
        gray_img = bg.convert('L')
    else:
        gray_img = raw_img.convert('L')

    # Detectar si fondo es blanco o negro
    arr = np.array(gray_img, dtype=np.float32)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    is_white_bg = np.mean(corners) > 127

    if is_white_bg:
        arr = 255.0 - arr

    # Si la imagen no es 14x14, centrar trazo y redimensionar
    if arr.shape != (14, 14):
        coords = np.argwhere(arr > 30)
        if coords.size > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            pad = int(max(ymax - ymin, xmax - xmin) * 0.15)
            h, w = arr.shape
            ymin = max(0, ymin - pad)
            xmin = max(0, xmin - pad)
            ymax = min(h, ymax + pad)
            xmax = min(w, xmax + pad)

            cropped = arr[ymin:ymax, xmin:xmax]
            c_img = Image.fromarray(cropped.astype(np.uint8))
            c_img.thumbnail((12, 12), Image.LANCZOS)
            final_img = Image.new('L', (14, 14), 0)
            paste_x = (14 - c_img.width) // 2
            paste_y = (14 - c_img.height) // 2
            final_img.paste(c_img, (paste_x, paste_y))
            arr = np.array(final_img, dtype=np.float32) / 255.0
        else:
            resized = gray_img.resize((14, 14), Image.LANCZOS)
            arr = np.array(resized, dtype=np.float32)
            arr = (255.0 - arr) / 255.0 if is_white_bg else arr / 255.0
    else:
        arr = arr / 255.0

    return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


def predict(image_input, top_k=5):
    """
    Predice la letra griega a partir de una imagen o vector de píxeles.

    Retorna un diccionario con:
      - label (1-24)
      - letter (carácter griego)
      - name (nombre en español/latín)
      - confidence (float 0.0 - 1.0)
      - top_k (lista de las mejores K predicciones)
    """
    if MODEL is None:
        raise RuntimeError("El modelo no ha sido cargado. Ejecuta load_model() primero.")

    tensor = preprocess_image(image_input)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)

        k = min(top_k, probabilities.size(1))
        top_conf, top_idx = torch.topk(probabilities, k, dim=1)

    pred_class = pred.item()
    pred_label = pred_class + 1  # 0-indexed a 1-indexed (1..24)

    top_candidates = []
    for i in range(k):
        idx = top_idx[0][i].item()
        lbl = idx + 1
        c = top_conf[0][i].item()
        top_candidates.append({
            'label': lbl,
            'letter': LABEL_TO_LETTER.get(lbl, '?'),
            'name': LETTER_NAMES.get(lbl, 'desconocido'),
            'confidence': float(c),
            'confidence_percent': f"{c * 100:.1f}%"
        })

    return {
        'label': pred_label,
        'letter': LABEL_TO_LETTER.get(pred_label, '?'),
        'name': LETTER_NAMES.get(pred_label, 'desconocido'),
        'confidence': float(conf.item()),
        'confidence_percent': f"{conf.item() * 100:.1f}%",
        'top_k': top_candidates
    }


def evaluate_test_samples(num_samples=24):
    """
    Ejecuta y muestra una evaluación en consola usando muestras del dataset test.csv.
    """
    test_csv = os.path.join(os.path.dirname(__file__), 'test.csv')
    if not os.path.exists(test_csv):
        print(f"No se encontro {test_csv} para ejecutar pruebas.")
        return

    import pandas as pd
    df = pd.read_csv(test_csv)

    print("=" * 70)
    print("EVALUACION DE MUESTRAS DEL DATASET DE PRUEBA (test.csv)")
    print("=" * 70)
    print(f"{'#':>3} | {'Real':<12} | {'Prediccion':<12} | {'Confianza':>10} | {'Resultado':<6}")
    print("-" * 70)

    correct = 0
    total = min(num_samples, len(df))

    # Tomar muestras representativas (una por letra si son 24)
    if num_samples == 24 and len(df['label'].unique()) == 24:
        sample_indices = [df[df['label'] == lbl].index[0] for lbl in range(1, 25)]
    else:
        sample_indices = list(range(total))

    for i, idx in enumerate(sample_indices, 1):
        row = df.loc[idx]
        true_label = int(row['label'])
        true_name = f"{LETTER_NAMES.get(true_label, '')} ({LABEL_TO_LETTER.get(true_label, '')})"

        pixel_data = row.drop('label').values.astype(np.float32)
        res = predict(pixel_data)

        pred_label = res['label']
        pred_name = f"{res['name']} ({res['letter']})"
        conf = res['confidence_percent']
        match = (true_label == pred_label)
        if match:
            correct += 1

        status = "OK" if match else "FALLO"
        print(f"{i:>3} | {true_name:<12} | {pred_name:<12} | {conf:>10} | {status:<6}")

    print("-" * 70)
    print(f"Aciertos: {correct}/{len(sample_indices)} ({correct / len(sample_indices):.1%})")
    print("=" * 70)


# ==============================================================================
# Rutas de la API REST (Sin HTML/CSS)
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    """Ruta principal: Informacion del estado de la API."""
    return jsonify({
        'status': 'online',
        'service': 'Clasificador de Letras Griegas (PyTorch CNN)',
        'model_loaded': MODEL is not None,
        'model_info': MODEL_INFO,
        'endpoints': {
            'GET /': 'Estado de la API y documentacion',
            'GET /model_info': 'Metadatos del modelo entrenado',
            'POST /predict': 'Clasificar imagen (JSON base64/pixeles o multipart/form-data)',
            'GET /test_sample': 'Clasificar una muestra de test.csv (parametro opcional ?label=1..24)'
        }
    })


@app.route("/model_info", methods=["GET"])
def get_model_info():
    """Retorna los metadatos del modelo entrenado."""
    if MODEL_INFO is None:
        return jsonify({'error': 'Modelo no cargado'}), 503
    return jsonify({
        'model_info': MODEL_INFO,
        'labels': {lbl: {'letter': LABEL_TO_LETTER[lbl], 'name': LETTER_NAMES[lbl]} for lbl in range(1, 25)}
    })


@app.route("/predict", methods=["POST"])
def api_predict():
    """
    Endpoint para clasificar una letra griega.
    Soporta:
      - JSON con {"image": "<base64>"}
      - JSON con {"pixels": [196 floats/ints]}
      - Archivo adjunto multipart/form-data con clave "file" o "image"
    """
    if MODEL is None:
        return jsonify({'error': 'Modelo no cargado en el servidor'}), 503

    try:
        image_input = None

        if request.is_json:
            data = request.get_json(silent=True) or {}
            if 'image' in data:
                image_input = data['image']
            elif 'pixels' in data:
                image_input = data['pixels']
            else:
                return jsonify({'error': 'Se requiere "image" (base64) o "pixels" (lista de 196 valores)'}), 400

        elif 'file' in request.files or 'image' in request.files:
            file = request.files.get('file') or request.files.get('image')
            image_input = file

        elif request.form.get('image'):
            image_input = request.form.get('image')

        else:
            return jsonify({'error': 'No se proporciono ninguna imagen o arreglo valido'}), 400

        result = predict(image_input)
        return jsonify(result)

    except Exception as err:
        return jsonify({'error': f"Fallo al procesar la prediccion: {str(err)}"}), 500


@app.route("/test_sample", methods=["GET"])
def api_test_sample():
    """Clasifica una muestra aleatoria o especifica del archivo test.csv."""
    if MODEL is None:
        return jsonify({'error': 'Modelo no cargado'}), 503

    test_csv = os.path.join(os.path.dirname(__file__), 'test.csv')
    if not os.path.exists(test_csv):
        return jsonify({'error': 'test.csv no encontrado'}), 404

    try:
        import pandas as pd
        df = pd.read_csv(test_csv)

        target_label = request.args.get('label', type=int)
        if target_label and 1 <= target_label <= 24:
            sub = df[df['label'] == target_label]
            row = sub.sample(n=1).iloc[0] if not sub.empty else df.sample(n=1).iloc[0]
        else:
            row = df.sample(n=1).iloc[0]

        true_label = int(row['label'])
        pixels = row.drop('label').values.astype(np.float32)

        pred_res = predict(pixels)
        pred_res['true_label'] = true_label
        pred_res['true_letter'] = LABEL_TO_LETTER.get(true_label, '?')
        pred_res['true_name'] = LETTER_NAMES.get(true_label, '?')
        pred_res['match'] = (true_label == pred_res['label'])

        return jsonify(pred_res)
    except Exception as err:
        return jsonify({'error': str(err)}), 500


# ==============================================================================
# Punto de Entrada
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clasificador de Letras Griegas (PyTorch CNN)")
    parser.add_argument("--test", action="store_true", help="Ejecuta evaluacion en muestras de test.csv y termina")
    parser.add_argument("--predict", type=str, help="Ruta de una imagen a clasificar")
    parser.add_argument("--serve", action="store_true", help="Inicia solo el servidor REST API")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Puerto del servidor (default: 5000)")
    args = parser.parse_args()

    # Cargar modelo
    loaded = load_model()
    if not loaded:
        print("ERROR: No se pudo cargar el modelo. Finalizando.")
        sys.exit(1)

    if args.predict:
        print(f"Clasificando archivo: {args.predict}")
        res = predict(args.predict)
        print(f"Letra: {res['letter']} ({res['name']}) | Confianza: {res['confidence_percent']}")
        print("Top 3:")
        for cand in res['top_k'][:3]:
            print(f"  - {cand['letter']} ({cand['name']}): {cand['confidence_percent']}")
        return

    if args.test:
        evaluate_test_samples(24)
        return

    if args.serve:
        print(f"Iniciando servidor REST API en http://localhost:{args.port}...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    # Comportamiento por defecto: Mostrar evaluacion de prueba y arrancar API
    evaluate_test_samples(24)
    print()
    print(f"Iniciando servidor REST API en http://localhost:{args.port}")
    print("Presiona Ctrl+C para detener el servidor.")
    print()
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
