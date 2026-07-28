"""Neural-network architectures for the MNIST detection project."""

from .model_factory import build_detector
from .mnist_detector import MNISTDetector
from .small_mnist_detector import SmallMNISTDetector
from .yolo_mnist_detector import YOLOLargeMNISTDetector, YOLOSmallMNISTDetector

__all__ = ["MNISTDetector", "SmallMNISTDetector", "YOLOSmallMNISTDetector", "YOLOLargeMNISTDetector", "build_detector"]
