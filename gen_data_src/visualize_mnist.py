import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def main() -> None:
    """Display a 4×4 grid of MNIST training images."""
    mnist = MNIST(
        root="./data",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))

    for index, axis in enumerate(axes.flat):
        image, label = mnist[index]
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
