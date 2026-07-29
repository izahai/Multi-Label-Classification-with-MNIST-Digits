from pathlib import Path
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def download_mnist(data_dir: str = "./data") -> None:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset = MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    print(f"Downloaded {len(train_dataset):,} training images")
    print(f"Downloaded {len(test_dataset):,} test images")
    print(f"Dataset location: {root.resolve()}")


if __name__ == "__main__":
    download_mnist()
