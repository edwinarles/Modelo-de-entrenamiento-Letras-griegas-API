"""
Entrenamiento de una CNN en PyTorch para clasificar letras griegas manuscritas.

El modelo es una red neuronal convolucional que toma imágenes de 14x14 píxeles
en escala de grises y clasifica entre 24 letras griegas.

Utiliza data augmentation extensiva para mejorar la generalización.

Uso:
    python train_model.py

Requisitos:
    - torch, torchvision
    - pandas, numpy
    - scikit-learn (para métricas)
    - train.csv y test.csv (generados por download_dataset.py)
"""

import os
import sys

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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json

# Configuración
# Force CPU - RTX 5060 (sm_120) requires newer PyTorch with CUDA 12.8+ support
DEVICE = torch.device('cpu')
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 14
NUM_CLASSES = 24

# Mapeo de labels a letras griegas
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


class GreekLetterDataset(Dataset):
    """Dataset de letras griegas desde CSV."""

    def __init__(self, csv_path, augment=False):
        df = pd.read_csv(csv_path)
        self.labels = df['label'].values - 1  # 0-indexed para PyTorch
        self.images = df.drop('label', axis=1).values.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(IMG_SIZE, IMG_SIZE)

        # Normalizar a [0, 1]
        img = img / 255.0

        # Invertir (fondo blanco -> fondo negro, como MNIST)
        img = 1.0 - img

        if self.augment:
            img = self._augment(img)

        # Agregar dimensión de canal
        img = torch.FloatTensor(img).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]

        return img, label

    def _augment(self, img):
        """Augmentaciones simples a nivel de numpy."""
        # Ruido aleatorio
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # Shift aleatorio (desplazar píxeles)
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-1, 2)
            shift_y = np.random.randint(-1, 2)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)

        # Escalar intensidad
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            img = np.clip(img * scale, 0, 1)

        return img.astype(np.float32)


class GreekLetterCNN(nn.Module):
    """
    Red Neuronal Convolucional para clasificar letras griegas.

    Arquitectura:
    - Conv2D(1, 32, 3) -> BatchNorm -> ReLU -> MaxPool
    - Conv2D(32, 64, 3) -> BatchNorm -> ReLU -> MaxPool
    - Conv2D(64, 128, 3) -> BatchNorm -> ReLU
    - Flatten -> FC(128*1*1, 256) -> Dropout -> FC(256, 24)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(GreekLetterCNN, self).__init__()

        self.features = nn.Sequential(
            # Bloque 1: 14x14 -> 6x6
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7

            # Bloque 2: 7x7 -> 2x2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 3x3

            # Bloque 3: 3x3 -> 1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer):
    """Entrena un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    print("=" * 60)
    print("ENTRENAMIENTO CNN - LETRAS GRIEGAS MANUSCRITAS")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, 'train.csv')
    test_csv = os.path.join(script_dir, 'test.csv')

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("ERROR: No se encontraron train.csv y test.csv")
        print("Ejecuta primero: python download_dataset.py")
        return

    # Cargar datasets
    print("Cargando datos...")
    train_dataset = GreekLetterDataset(train_csv, augment=True)
    test_dataset = GreekLetterDataset(test_csv, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"Muestras de entrenamiento: {len(train_dataset)}")
    print(f"Muestras de prueba: {len(test_dataset)}")
    print()

    # Crear modelo
    model = GreekLetterCNN(NUM_CLASSES).to(DEVICE)
    print("Arquitectura del modelo:")
    print(model)
    print()

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print()

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,
                                                      factor=0.5)

    # Entrenamiento
    best_test_acc = 0
    print("Iniciando entrenamiento...")
    print("-" * 60)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)

        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_path = os.path.join(script_dir, 'greek_letter_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'img_size': IMG_SIZE,
                'label_to_letter': LABEL_TO_LETTER,
                'letter_names': LETTER_NAMES,
                'test_accuracy': test_acc,
                'epoch': epoch,
            }, model_path)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {test_loss:>9.4f} | {test_acc:>7.1%}"
              + (" *" if test_acc == best_test_acc else ""))

    print("-" * 60)
    print(f"Mejor accuracy en test: {best_test_acc:.1%}")
    print()

    # Evaluación final
    print("Evaluación final con el mejor modelo...")
    checkpoint = torch.load(os.path.join(script_dir, 'greek_letter_model.pth'),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, preds, labels = evaluate(model, test_loader, criterion)

    # Reporte de clasificación
    target_names = [f"{LETTER_NAMES[i+1]} ({i+1})" for i in range(NUM_CLASSES)]
    print()
    print("Reporte de clasificacion:")
    print(classification_report(labels, preds, target_names=target_names, zero_division=0))

    # Guardar metadatos del modelo para el servidor web
    metadata = {
        'num_classes': NUM_CLASSES,
        'img_size': IMG_SIZE,
        'label_to_letter': {str(k): v for k, v in LABEL_TO_LETTER.items()},
        'letter_names': {str(k): v for k, v in LETTER_NAMES.items()},
        'test_accuracy': float(final_acc),
    }
    metadata_path = os.path.join(script_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: greek_letter_model.pth")
    print(f"Metadata guardada en: model_metadata.json")
    print()
    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()
