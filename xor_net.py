#!/usr/bin/env python3
"""
XOR Neural Network - Train a network to recognize XOR function using PyTorch.
"""

import argparse
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim


class XORNet(nn.Module):
    """Simple 2-layer neural network for XOR function."""

    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden = nn.Linear(2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x


def xor_label(a: float, b: float) -> float:
    """Compute XOR label for two float inputs using 0.5 threshold."""
    a_bool = a >= 0.5
    b_bool = b >= 0.5
    return 1.0 if a_bool != b_bool else 0.0


def generate_data(num_samples: int, seed: int) -> list[dict]:
    """Generate random training/test data for XOR function."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        a = random.random()
        b = random.random()
        label = xor_label(a, b)
        data.append({"inputs": [a, b], "label": label})
    return data


def save_data(data: list[dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filepath}")


def load_data(filepath: str) -> list[dict]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def data_to_tensors(data: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert data list to PyTorch tensors."""
    inputs = torch.tensor([[d["inputs"][0], d["inputs"][1]] for d in data], dtype=torch.float32, device=device)
    labels = torch.tensor([[d["label"]] for d in data], dtype=torch.float32, device=device)
    return inputs, labels


def train(model: XORNet, data_path: str, weights_path: str, epochs: int = 1000, lr: float = 1.0, device: torch.device = None):
    """Train the model and save weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    data = load_data(data_path)
    inputs, labels = data_to_tensors(data, device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print(f"Training on {device} with {len(data)} samples for {epochs} epochs...")

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")


def test(model: XORNet, data_path: str, weights_path: str):
    """Test the model and report error rate with GPU/CPU benchmarks."""
    data = load_data(data_path)

    # Test on GPU first (if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = XORNet()
        model_gpu.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model_gpu = model_gpu.to(device)
        model_gpu.eval()

        inputs, labels = data_to_tensors(data, device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(inputs)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model_gpu(inputs)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        predictions = (outputs >= 0.5).float()
        correct = (predictions == labels).sum().item()
        error_rate = 1.0 - (correct / len(data))

        print(f"\n=== GPU Benchmark ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Inference time: {gpu_time * 1000:.4f} ms")
        print(f"Samples: {len(data)}")
        print(f"Correct: {int(correct)}/{len(data)}")
        print(f"Error rate: {error_rate * 100:.2f}%")
    else:
        print("\nGPU not available, skipping GPU benchmark.")

    # Test on CPU
    device = torch.device("cpu")
    model_cpu = XORNet()
    model_cpu.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model_cpu = model_cpu.to(device)
    model_cpu.eval()

    inputs, labels = data_to_tensors(data, device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(inputs)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model_cpu(inputs)
    cpu_time = time.perf_counter() - start_time

    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels).sum().item()
    error_rate = 1.0 - (correct / len(data))

    print(f"\n=== CPU Benchmark ===")
    print(f"Inference time: {cpu_time * 1000:.4f} ms")
    print(f"Samples: {len(data)}")
    print(f"Correct: {int(correct)}/{len(data)}")
    print(f"Error rate: {error_rate * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="XOR Neural Network - Train and test a network to recognize XOR function")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate random training/test data")
    gen_parser.add_argument("--seed", type=int, required=True, help="Random seed for data generation")
    gen_parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    gen_parser.add_argument("--output", type=str, default="data.json", help="Output file path (default: data.json)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the network and save weights")
    train_parser.add_argument("--data", type=str, default="data.json", help="Training data file (default: data.json)")
    train_parser.add_argument("--weights", type=str, default="weights.pt", help="Output weights file (default: weights.pt)")
    train_parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    train_parser.add_argument("--lr", type=float, default=1.0, help="Learning rate (default: 1.0)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the network and report error rate")
    test_parser.add_argument("--data", type=str, default="data.json", help="Test data file (default: data.json)")
    test_parser.add_argument("--weights", type=str, default="weights.pt", help="Weights file to load (default: weights.pt)")

    args = parser.parse_args()

    if args.command == "generate":
        data = generate_data(args.samples, args.seed)
        save_data(data, args.output)
    elif args.command == "train":
        model = XORNet()
        train(model, args.data, args.weights, args.epochs, args.lr)
    elif args.command == "test":
        model = XORNet()
        test(model, args.data, args.weights)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
