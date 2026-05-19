"""Training scaffold for the Task 2 MNIST digit classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import torch
from torch import nn

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNIST_DATA_DIR = TASK_ROOT / "data"


def download_mnist_dataset(data_dir: Path = DEFAULT_MNIST_DATA_DIR) -> Path:
    """Download torchvision MNIST into the Task 2 data directory."""
    import torchvision

    data_dir.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return data_dir / "MNIST"


class MNISTClassifier(nn.Module): # Implemented
    """Small PyTorch classifier scaffold for 28x28 MNIST crops."""

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        # TODO(student): fill in your custom model architectures
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, num_classes)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        # raise NotImplementedError("MNIST classifier model logic not implemented!")

    def forward(self, inputs):
        # TODO(student): fill in your forward process according to your model
        x = self.net(inputs)
        return x


def select_training_device(torch_module) -> str: # Implemented
    """Return the best available device string: 'cuda', 'mps', or 'cpu'.

    The checks are written defensively because older or custom builds of
    PyTorch may not expose all attributes (for example, `backends.mps`).
    """
    try:
        if getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        backends = getattr(torch_module, "backends", None)
        if backends is not None and getattr(backends, "mps", None) is not None and backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def train_mnist_classifier(dataset_dir: Path, output_path: Path) -> Path: # Implemented
    
    from torch.utils.data import DataLoader, random_split
    import torchvision
    # TODO(student): Train the MNIST digit classifier used by model.py.
    # device = select_training_device(torch)
    # move the model and each batch to device
    # read training images and labels from dataset_dir
    # split examples into training and validation sets
    # preprocess every image the same way model.preprocess_mnist_crop does
    # model = MNISTClassifier()
    # choose loss function, optimizer, batch size, and number of epochs
    # train until validation accuracy is stable
    # save the trained model weights or serialized estimator to output_path
    # return output_path
    device = select_training_device(torch)
    print(f"Selected training device: {device}")
    torch_device = torch.device(device)
    model = MNISTClassifier().to(torch_device)
    data = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=False, transform=torchvision.transforms.ToTensor())
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(torch_device), labels.to(torch_device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(torch_device), labels.to(torch_device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), output_path)
    return output_path
    
    raise NotImplementedError("MNIST training is not implemented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task 2 MNIST digit classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MNIST_DATA_DIR , help="Directory containing labeled MNIST board crops.")
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "models" / "mnist_classifier.npz", help="Where to save the trained classifier.")
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST into tasks/task2-detector/data/MNIST before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        print(f"Dataset directory {args.dataset_dir} does not exist. Use --download-mnist to download the MNIST dataset.")
        return
    if not args.output.parent.exists():
        os.makedirs(args.output.parent, exist_ok=True)
    if args.download_mnist:
        dataset_path = download_mnist_dataset(DEFAULT_MNIST_DATA_DIR)
        print(f"Downloaded MNIST dataset to: {dataset_path}")
        return

    output_path = train_mnist_classifier(args.dataset_dir, args.output)
    print(f"Saved MNIST classifier to: {output_path}")


if __name__ == "__main__":
    main()
