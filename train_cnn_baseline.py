import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_ROOT = "./data"
OUTPUT_DIR = "./trained_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TEST_SIZE = 0.20
RANDOM_STATE = 42

BREEDS = {
    "beagle": "Beagle",
    "boxer": "Boxer",
    "chihuahua": "Chihuahua",
    "pug": "Pug",
    "samoyed": "Samoyed",
    "shiba_inu": "ShibaInu",
    "great_pyrenees": "GreatPyrenees",
    "german_shorthaired": "GermanShorthairedPointer",
    "staffordshire_bull_terrier": "StaffordshireBullTerrier",
    "yorkshire_terrier": "YorkshireTerrier",
    "pomeranian": "Pomeranian",
    "basset_hound": "BassetHound",
    "saint_bernard": "SaintBernard",
}

def norm(name):
    return name.lower().replace(" ", "_")

class LabeledSubset(Dataset):
    def __init__(self, base, indices, labels):
        self.base = base
        self.indices = [int(x) for x in indices]
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image, _ = self.base[idx]
        return image, int(self.labels[idx])

def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    ds = OxfordIIITPet(
        root=DATA_ROOT,
        download=True,
        target_types="category",
        transform=transform,
    )

    breed_names = sorted(BREEDS)
    breed_to_idx = {b: i for i, b in enumerate(breed_names)}

    selected = []
    labels = np.full(len(ds), -1, dtype=np.int64)

    for i in range(len(ds)):
        class_idx = ds._labels[i]
        breed = norm(ds.classes[class_idx])

        if breed in breed_to_idx:
            selected.append(i)
            labels[i] = breed_to_idx[breed]

    selected = np.asarray(selected, dtype=np.int64)

    print(f"Selected {len(selected)} images from 13 breeds.")
    print("Class mapping:")
    for b in breed_names:
        print(f"  {breed_to_idx[b]:2d} -> {BREEDS[b]}")

    print(f"Label range: {labels[selected].min()} to {labels[selected].max()}")

    return ds, selected, labels, breed_to_idx

def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 13)
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    top3_hits = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        pred = outputs.argmax(1)

        ys.extend(labels.cpu().numpy())
        preds.extend(pred.cpu().numpy())

        top3 = outputs.topk(3, dim=1).indices
        top3_hits += (top3 == labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)

    return accuracy_score(ys, preds), top3_hits / total

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ds, selected, labels, breed_to_idx = load_data()

    train_idx, test_idx = train_test_split(
        selected,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels[selected],
    )

    train_set = LabeledSubset(ds, train_idx, labels)
    test_set = LabeledSubset(ds, test_idx, labels)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    print(f"Training images: {len(train_set)}")
    print(f"Testing images:  {len(test_set)}")

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    print("\nTraining CNN baseline...\n")

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        top1, top3 = evaluate(model, test_loader)
        losses.append(loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| loss: {loss:.4f} "
            f"| top-1: {top1:.2%} "
            f"| top-3: {top3:.2%}"
        )

    top1, top3 = evaluate(model, test_loader)

    print("\n=== CNN BASELINE RESULT ===")
    print(f"Images: {len(selected)}")
    print(f"Train:  {len(train_set)}")
    print(f"Test:   {len(test_set)}")
    print(f"Top-1:  {top1:.2%}")
    print(f"Top-3:  {top3:.2%}")

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "cnn_baseline.pt")
    )

    with open(
        os.path.join(OUTPUT_DIR, "cnn_label_mapping.pkl"), "wb"
    ) as f:
        pickle.dump(breed_to_idx, f)

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("ResNet-18 CNN baseline - training loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_loss_curve.png"))
    plt.close()

    print(f"\nSaved CNN model and loss curve to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
