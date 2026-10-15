"""
Script para descargar el dataset de letras griegas manuscritas de Kaggle.

Dataset: Classification of Handwritten Greek Letters
URL: https://www.kaggle.com/datasets/katianakontolati/classification-of-handwritten-greek-letters

Contiene 24 letras griegas minusculas como imagenes 14x14 en escala de grises.
- train.csv: 500*24 muestras (500 por letra), 196 features + 1 label
- test.csv: 96 muestras (4 por letra), 196 features + 1 label

Etiquetas:
α=>1, β=>2, γ=>3, δ=>4, ε=>5, ζ=>6, η=>7, θ=>8, ι=>9, κ=>10,
λ=>11, μ=>12, ν=>13, ξ=>14, ο=>15, π=>16, ρ=>17, σ=>18, τ=>19, υ=>20,
φ=>21, χ=>22, ψ=>23, ω=>24
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def download_kaggle_dataset():
    """Descarga el dataset usando opendatasets (requiere credenciales de Kaggle)."""
    try:
        import opendatasets as od
    except ImportError:
        print("Instalando opendatasets...")
        os.system(f"{sys.executable} -m pip install opendatasets")
        import opendatasets as od

    dataset_url = "https://www.kaggle.com/datasets/katianakontolati/classification-of-handwritten-greek-letters"
    download_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("DESCARGA DEL DATASET DE LETRAS GRIEGAS")
    print("=" * 60)
    print()
    print("Se necesitan credenciales de Kaggle.")
    print("Puedes obtenerlas en: https://www.kaggle.com/settings")
    print("  1. Ve a tu perfil -> Settings -> API")
    print("  2. Haz clic en 'Create New Token'")
    print("  3. Se descargará un archivo kaggle.json")
    print()

    od.download(dataset_url, data_dir=download_dir)

    # Mover archivos CSV al directorio principal
    dataset_dir = os.path.join(download_dir, "classification-of-handwritten-greek-letters")
    if os.path.exists(dataset_dir):
        for fname in ["train.csv", "test.csv"]:
            src = os.path.join(dataset_dir, fname)
            dst = os.path.join(download_dir, fname)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"  Copiado: {fname}")

    print()
    print("¡Dataset descargado exitosamente!")
    print(f"  - train.csv: datos de entrenamiento")
    print(f"  - test.csv: datos de prueba")


def generate_synthetic_dataset():
    """
    Genera un dataset sintético de letras griegas usando fuentes del sistema.
    Esto permite crear una gran cantidad de datos sin necesidad de Kaggle.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    import random

    print("=" * 60)
    print("GENERANDO DATASET SINTETICO DE LETRAS GRIEGAS")
    print("=" * 60)

    # Las 24 letras griegas minusculas
    greek_letters = [
        '\u03b1', '\u03b2', '\u03b3', '\u03b4', '\u03b5', '\u03b6', '\u03b7', '\u03b8',
        '\u03b9', '\u03ba', '\u03bb', '\u03bc', '\u03bd', '\u03be', '\u03bf', '\u03c0',
        '\u03c1', '\u03c3', '\u03c4', '\u03c5', '\u03c6', '\u03c7', '\u03c8', '\u03c9'
    ]

    # Nombres ASCII de las letras griegas para imprimir en consola Windows
    greek_names = [
        'alfa', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
        'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
    ]

    img_size = 14
    render_size = 64  # Renderizar más grande y luego reducir

    # Buscar fuentes que soporten caracteres griegos
    font_paths = []
    system_font_dirs = [
        "C:/Windows/Fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
    ]

    # Fuentes que típicamente soportan caracteres griegos
    preferred_fonts = [
        "arial.ttf", "times.ttf", "cour.ttf", "calibri.ttf",
        "cambria.ttc", "consola.ttf", "georgia.ttf", "tahoma.ttf",
        "verdana.ttf", "trebuc.ttf", "segoeui.ttf", "comic.ttf",
        "impact.ttf", "palai.ttf", "BOOKOS.TTF", "GARABD.TTF",
        "ariali.ttf", "timesi.ttf", "timesbd.ttf", "arialbd.ttf",
        "calibrib.ttf", "calibrii.ttf", "georgiab.ttf", "georgiaz.ttf",
        "tahomabd.ttf", "verdanab.ttf", "verdanai.ttf",
    ]

    for font_dir in system_font_dirs:
        if os.path.exists(font_dir):
            for fname in preferred_fonts:
                full_path = os.path.join(font_dir, fname)
                if os.path.exists(full_path):
                    font_paths.append(full_path)

    if not font_paths:
        # Buscar cualquier .ttf disponible
        for font_dir in system_font_dirs:
            if os.path.exists(font_dir):
                for f in os.listdir(font_dir):
                    if f.lower().endswith(('.ttf', '.otf')):
                        font_paths.append(os.path.join(font_dir, f))
                        if len(font_paths) >= 10:
                            break

    if not font_paths:
        print("ERROR: No se encontraron fuentes del sistema.")
        print("Intenta descargar el dataset de Kaggle con la opción 1.")
        return

    print(f"Encontradas {len(font_paths)} fuentes del sistema")

    # Filtrar fuentes que realmente pueden renderizar caracteres griegos
    valid_fonts = []
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, 40)
            img = Image.new('L', (64, 64), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), 'α', font=font, fill=0)
            arr = np.array(img)
            if arr.min() < 200:  # La letra se renderizó
                valid_fonts.append(fp)
        except Exception:
            pass

    if not valid_fonts:
        print("ERROR: Ninguna fuente puede renderizar caracteres griegos.")
        return

    print(f"Fuentes válidas para griego: {len(valid_fonts)}")

    def render_letter(letter, font_path, font_size, offset_x=0, offset_y=0,
                      rotation=0, noise_level=0, thickness=0):
        """Renderiza una letra griega como imagen 14x14 en escala de grises."""
        img = Image.new('L', (render_size, render_size), 255)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # Obtener tamaño del texto
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Centrar el texto con offset
        x = (render_size - text_w) // 2 + offset_x - bbox[0]
        y = (render_size - text_h) // 2 + offset_y - bbox[1]

        draw.text((x, y), letter, font=font, fill=0)

        # Aplicar engrosamiento simulado
        if thickness > 0:
            from PIL import ImageFilter
            for _ in range(thickness):
                img = img.filter(ImageFilter.MinFilter(3))

        # Rotar
        if rotation != 0:
            img = img.rotate(rotation, fillcolor=255, resample=Image.BICUBIC)

        # Reducir a 14x14
        img = img.resize((img_size, img_size), Image.LANCZOS)

        # Agregar ruido
        arr = np.array(img, dtype=np.float32)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, arr.shape)
            arr = np.clip(arr + noise, 0, 255)

        return arr.astype(np.uint8)

    # Generar muestras con data augmentation
    samples_per_letter_train = 500  
    samples_per_letter_test = 20    # 20 muestras por letra para test

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    font_sizes = list(range(30, 52, 2))

    for label_idx, letter in enumerate(greek_letters, 1):
        letter_name = greek_names[label_idx - 1]
        print(f"  Generando letra {letter_name} (label {label_idx})...", end=" ")

        # Datos de entrenamiento
        for i in range(samples_per_letter_train):
            font_path = random.choice(valid_fonts)
            font_size = random.choice(font_sizes)
            offset_x = random.randint(-6, 6)
            offset_y = random.randint(-6, 6)
            rotation = random.uniform(-20, 20)
            noise_level = random.uniform(0, 25)
            thickness = random.choice([0, 0, 0, 1, 1, 2])

            img = render_letter(letter, font_path, font_size,
                              offset_x, offset_y, rotation,
                              noise_level, thickness)
            train_data.append(img.flatten())
            train_labels.append(label_idx)

        # Datos de test (menos augmentation para ser más "limpios")
        for i in range(samples_per_letter_test):
            font_path = random.choice(valid_fonts)
            font_size = random.choice(font_sizes)
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)
            rotation = random.uniform(-10, 10)
            noise_level = random.uniform(0, 10)

            img = render_letter(letter, font_path, font_size,
                              offset_x, offset_y, rotation,
                              noise_level, 0)
            test_data.append(img.flatten())
            test_labels.append(label_idx)

        print(f"OK ({samples_per_letter_train} train, {samples_per_letter_test} test)")

    # Crear DataFrames
    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_labels

    test_df = pd.DataFrame(test_data)
    test_df['label'] = test_labels

    # Guardar CSVs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_df.to_csv(os.path.join(script_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(script_dir, 'test.csv'), index=False)

    print()
    print(f"Dataset generado exitosamente!")
    print(f"  - train.csv: {len(train_df)} muestras ({samples_per_letter_train} por letra)")
    print(f"  - test.csv: {len(test_df)} muestras ({samples_per_letter_test} por letra)")
    print(f"  - Tamaño de imagen: {img_size}x{img_size} pixels")
    print(f"  - Número de clases: {len(greek_letters)}")


if __name__ == "__main__":
    print()
    print("Selecciona método de obtención del dataset:")
    print("  1. Descargar de Kaggle (requiere cuenta)")
    print("  2. Generar dataset sintético (recomendado, no requiere cuenta)")
    print()

    choice = input("Opción (1 o 2): ").strip()

    if choice == "1":
        download_kaggle_dataset()
    elif choice == "2":
        generate_synthetic_dataset()
    else:
        print("Opción no válida. Generando dataset sintético por defecto...")
        generate_synthetic_dataset()
