"""
Testa o loader do MNIST.
Usa um FakeMNIST sintético (sem download) via monkeypatch de
torchvision.datasets.MNIST.
"""
import numpy as np
import torch
import torchvision
import pytest
from PIL import Image

from datasets.mnist import MNISTDataset


N_FAKE = 40


class FakeMNIST:
    def __init__(self, root, train=True, download=True):
        self.data = [
            Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
            for _ in range(N_FAKE)
        ]
        self.targets = torch.tensor([i % 10 for i in range(N_FAKE)])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


@pytest.fixture
def mnist_root(monkeypatch, tmp_path):
    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)
    return str(tmp_path)


def test_dataset_loads(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    assert len(ds) == N_FAKE


def test_getitem_keys(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    item = ds[0]
    assert "image" in item
    assert "filename" in item
    assert "path" in item
    assert "digit" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)
    assert isinstance(item["digit"], int)


def test_getitem_saves_png(mnist_root):
    import os
    ds = MNISTDataset(root=mnist_root)
    item = ds[0]
    assert os.path.exists(item["path"])


def test_sample_stratified_size(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    subset = ds.sample(n=20, strategy="stratified", seed=42)
    assert len(subset) > 0
    assert len(subset) <= 20


def test_sample_stratified_balanced(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    subset = ds.sample(n=20, strategy="stratified", seed=42)
    digits = [subset[i]["digit"] for i in range(len(subset))]
    assert len(set(digits)) > 1


def test_sample_random_reproducible(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    s1 = ds.sample(n=10, strategy="random", seed=42)
    s2 = ds.sample(n=10, strategy="random", seed=42)
    assert [s1[i]["filename"] for i in range(len(s1))] == [s2[i]["filename"] for i in range(len(s2))]


def test_sample_unknown_strategy(mnist_root):
    ds = MNISTDataset(root=mnist_root)
    with pytest.raises(ValueError, match="Estratégia desconhecida"):
        ds.sample(n=10, strategy="uniform_bins")
