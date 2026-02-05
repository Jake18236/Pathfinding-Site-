#!/usr/bin/env python3
"""
Generate training prediction data for Cross-Entropy Loss visualization using CIFAR-10.

This script trains a small CNN on CIFAR-10 to show gradual convergence of predictions
over many epochs.

Output: static/json/cross-entropy-cifar10.json
"""

import json
import base64
import io
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image


# Configuration - tuned for visible but convergent learning
NUM_SAMPLES = 10  # One per class
SNAPSHOT_EPOCHS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
TOTAL_EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.01
TRAIN_SUBSET_SIZE = 15000  # Use 15k samples (CIFAR-10 is harder)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class names
CLASS_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Output path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'static', 'json', 'cross-entropy-cifar10.json')


class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10 - deliberately limited capacity for visible learning.
    """

    def __init__(self):
        super(SmallCNN, self).__init__()
        # Two conv layers with small filters
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pooling layers: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        """Get softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def image_to_base64(img_array):
    """Convert numpy array (32x32x3 RGB) to base64 PNG string."""
    # img_array should be in CHW format, convert to HWC for PIL
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))

    # Ensure it's in 0-255 range
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def select_samples(dataset, num_per_class=1):
    """
    Select samples from dataset, one per class.
    Returns list of (image, label, index) tuples.
    """
    samples = []
    class_counts = {i: 0 for i in range(10)}

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if class_counts[label] < num_per_class:
            samples.append({
                'image': img,
                'label': label,
                'index': idx
            })
            class_counts[label] += 1

        if all(c >= num_per_class for c in class_counts.values()):
            break

    # Sort by label for nice display
    samples.sort(key=lambda x: x['label'])
    return samples


def get_predictions(model, samples, device):
    """Get model predictions for all samples."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for sample in samples:
            img = sample['image'].unsqueeze(0).to(device)
            probs = model.predict_proba(img)
            predictions.append(probs.cpu().numpy()[0])

    return predictions


def calculate_loss(true_label, predictions, epsilon=1e-15):
    """Calculate cross-entropy loss for a single sample."""
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.log(predictions[true_label])


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total


def main():
    print(f"Using device: {DEVICE}")
    print(f"Training CIFAR-10 with:")
    print(f"  - Small CNN (2 conv layers, 64-dim hidden)")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Training subset: {TRAIN_SUBSET_SIZE} samples")
    print()

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    # Use only a subset of training data
    indices = torch.randperm(len(full_train_dataset))[:TRAIN_SUBSET_SIZE]
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Select samples to track (one per class) - from full dataset for variety
    print(f"Selecting {NUM_SAMPLES} samples to track...")
    samples = select_samples(full_train_dataset, num_per_class=1)

    # Prepare sample data for output
    sample_data = []
    for i, sample in enumerate(samples):
        # Convert image tensor to numpy for base64 encoding
        img_np = sample['image'].numpy()

        sample_data.append({
            'index': i,
            'trueLabel': int(sample['label']),
            'imageBase64': image_to_base64(img_np),
            'snapshots': []
        })

    # Initialize model
    model = SmallCNN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Training loop with snapshot capture
    print(f"Training for {TOTAL_EPOCHS} epochs...")

    for epoch in range(TOTAL_EPOCHS + 1):
        # Capture snapshot at specified epochs
        if epoch in SNAPSHOT_EPOCHS:
            print(f"  Capturing snapshot at epoch {epoch}...")
            predictions = get_predictions(model, samples, DEVICE)

            for i, (sample, preds) in enumerate(zip(samples, predictions)):
                true_label = sample['label']
                loss = calculate_loss(true_label, preds)
                predicted_class = int(np.argmax(preds))

                snapshot = {
                    'epoch': epoch,
                    'predictions': [float(p) for p in preds],
                    'loss': float(loss),
                    'predictedClass': predicted_class,
                    'correct': predicted_class == true_label
                }
                sample_data[i]['snapshots'].append(snapshot)

        # Train one epoch (skip epoch 0 - that's before training)
        if epoch < TOTAL_EPOCHS:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)

            if epoch % 5 == 0 or epoch == TOTAL_EPOCHS - 1:
                test_loss, test_acc = test_model(model, test_loader, DEVICE)
                print(f"  Epoch {epoch + 1}/{TOTAL_EPOCHS}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Test Acc: {test_acc:.4f}")

    # Get final test accuracy
    final_test_loss, final_test_acc = test_model(model, test_loader, DEVICE)
    print(f"\nFinal Test Accuracy: {final_test_acc * 100:.2f}%")

    # Prepare output JSON
    output = {
        'metadata': {
            'dataset': 'cifar10',
            'numSamples': NUM_SAMPLES,
            'numClasses': 10,
            'totalEpochs': TOTAL_EPOCHS,
            'snapshotEpochs': SNAPSHOT_EPOCHS,
            'numSnapshots': len(SNAPSHOT_EPOCHS),
            'finalTestAccuracy': round(final_test_acc * 100, 2),
            'generated': datetime.now().isoformat(),
            'classLabels': CLASS_LABELS
        },
        'samples': sample_data
    }

    # Write to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {OUTPUT_PATH}")

    # Print summary
    print("\nSample summary:")
    for sample in sample_data:
        first_loss = sample['snapshots'][0]['loss']
        last_loss = sample['snapshots'][-1]['loss']
        last_pred = sample['snapshots'][-1]['predictedClass']
        true_label = sample['trueLabel']
        print(f"  {CLASS_LABELS[true_label]}: "
              f"Loss {first_loss:.3f} -> {last_loss:.3f}, "
              f"Final prediction: {CLASS_LABELS[last_pred]} "
              f"({'correct' if last_pred == true_label else 'WRONG'})")


if __name__ == '__main__':
    main()
