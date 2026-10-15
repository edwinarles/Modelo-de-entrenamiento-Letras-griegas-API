"""
Evaluación y visualización de métricas del modelo de letras griegas.

Este script carga el modelo entrenado (greek_letter_model.pth) y evalúa
su rendimiento sobre el conjunto de test (o train), mostrando:
1. Resumen global (Accuracy, Loss, Total de aciertos/errores)
2. Tabla detallada de métricas por letra (Precisión, Recall, F1-Score)
3. Desglose detallado de confusiones/errores
4. Gráfico de Matriz de Confusión guardado como imagen ('confusion_matrix.png')

Uso:
    python evaluate_metrics.py
    python evaluate_metrics.py --split train
    python evaluate_metrics.py --no-plot
"""

import os
import sys
import argparse

# Configurar salida UTF-8 en Windows para símbolos griegos
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_model import (
    GreekLetterDataset,
    GreekLetterCNN,
    LABEL_TO_LETTER,
    LETTER_NAMES,
    NUM_CLASSES,
    DEVICE,
)


def evaluate_dataset(model, csv_path, batch_size=32):
    """Evalúa el modelo sobre un archivo CSV y retorna métricas detalladas."""
    dataset = GreekLetterDataset(csv_path, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return avg_loss, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    """Genera y guarda la matriz de confusión usando matplotlib."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión - Clasificación de Letras Griegas", fontsize=16, pad=15)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, class_names, fontsize=9)

        # Anotar los valores en cada celda
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0:
                    plt.text(
                        j,
                        i,
                        f"{val}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if val > thresh else "black",
                        fontsize=8,
                    )

        plt.ylabel("Etiqueta Real (Ground Truth)", fontsize=12)
        plt.xlabel("Predicción del Modelo", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"Nota: No se pudo generar el gráfico de la matriz de confusión: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluar métricas del modelo de letras griegas.")
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Conjunto de datos a evaluar ('test' o 'train', por defecto 'test')",
    )
    parser.add_argument(
        "--model",
        default="greek_letter_model.pth",
        help="Ruta al archivo del modelo .pth",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Desactivar la generación de la imagen de matriz de confusión",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    csv_file = "test.csv" if args.split == "test" else "train.csv"
    csv_path = os.path.join(script_dir, csv_file)

    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en: {model_path}")
        print("Ejecuta primero 'python train_model.py' para entrenarlo.")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo de datos en: {csv_path}")
        print("Ejecuta primero 'python download_dataset.py'.")
        sys.exit(1)

    print("=" * 75)
    print(f" EVALUACIÓN DE MÉTRICAS - DATASET: {csv_file.upper()}")
    print("=" * 75)

    # Cargar modelo
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = GreekLetterCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch_info = checkpoint.get("epoch", "N/A")
    print(f"Modelo cargado: {args.model} (Época guardada: {epoch_info})")
    print(f"Evaluando sobre: {csv_path} ...\n")

    # Evaluación
    loss, labels, preds, probs = evaluate_dataset(model, csv_path)
    total_samples = len(labels)
    correct_samples = int(np.sum(labels == preds))
    error_samples = total_samples - correct_samples
    overall_acc = accuracy_score(labels, preds)

    # 1. Resumen Global
    print("+" + "-" * 73 + "+")
    print(f"| {'RESUMEN GLOBAL':^71} |")
    print("+" + "-" * 73 + "+")
    print(f"| Muestras Evaluadas: {total_samples:<15} Aciertos: {correct_samples:<10} Errores: {error_samples:<6} |")
    print(f"| Accuracy Global:    {overall_acc * 100:>6.2f}%         Pérdida (Cross-Entropy Loss): {loss:<10.4f} |")
    print("+" + "-" * 73 + "+\n")

    # 2. Reporte Detallado por Letra
    target_names = [f"{LABEL_TO_LETTER[i+1]} ({LETTER_NAMES[i+1]})" for i in range(NUM_CLASSES)]
    report_dict = classification_report(
        labels, preds, target_names=target_names, output_dict=True, zero_division=0
    )

    print("+" + "-" * 73 + "+")
    print(f"| {'MÉTRICAS POR CATEGORÍA':^71} |")
    print("+" + "-" * 73 + "+")
    print(f"| {'Clase':<16} | {'Precisión':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Muestras':>10} |")
    print("+" + "-" * 73 + "+")

    for i in range(NUM_CLASSES):
        name = target_names[i]
        metrics = report_dict[name]
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1-score"]
        supp = int(metrics["support"])
        print(f"| {name:<16} | {prec:>9.4f}  | {rec:>9.4f}  | {f1:>9.4f}  | {supp:>10} |")

    print("+" + "-" * 73 + "+")
    macro_avg = report_dict["macro avg"]
    print(
        f"| {'Macro Avg':<16} | {macro_avg['precision']:>9.4f}  | {macro_avg['recall']:>9.4f}  | {macro_avg['f1-score']:>9.4f}  | {total_samples:>10} |"
    )
    print("+" + "-" * 73 + "+\n")

    # 3. Desglose de Errores y Confusiones
    cm = confusion_matrix(labels, preds)
    errors = []
    for actual_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            if actual_idx != pred_idx and cm[actual_idx, pred_idx] > 0:
                count = cm[actual_idx, pred_idx]
                actual_name = f"{LABEL_TO_LETTER[actual_idx+1]} ({LETTER_NAMES[actual_idx+1]})"
                pred_name = f"{LABEL_TO_LETTER[pred_idx+1]} ({LETTER_NAMES[pred_idx+1]})"
                errors.append((actual_name, pred_name, count))

    print("+" + "-" * 73 + "+")
    print(f"| {'DESGLOSE DE CONFUSIONES / ERRORES':^71} |")
    print("+" + "-" * 73 + "+")
    if not errors:
        print(f"| {'¡No se registraron errores! Clasificación 100% perfecta.':^71} |")
    else:
        print(f"| {'Letra Real':<22} -> {'Confundida con':<25} | {'Veces':>15} |")
        print("+" + "-" * 73 + "+")
        for actual_name, pred_name, count in sorted(errors, key=lambda x: -x[2]):
            print(f"| {actual_name:<22} -> {pred_name:<25} | {count:>15} |")
    print("+" + "-" * 73 + "+\n")

    # 4. Generación de Imagen de Matriz de Confusión
    if not args.no_plot:
        class_labels = [f"{LABEL_TO_LETTER[i+1]} {LETTER_NAMES[i+1]}" for i in range(NUM_CLASSES)]
        img_name = f"confusion_matrix_{args.split}.png"
        img_path = os.path.join(script_dir, img_name)
        if plot_confusion_matrix(cm, class_labels, img_path):
            print(f"-> Gráfico de matriz de confusión guardado en: {img_name}")

    print("=" * 75)
    print(" Evaluación completada con éxito.")
    print("=" * 75)


if __name__ == "__main__":
    main()
