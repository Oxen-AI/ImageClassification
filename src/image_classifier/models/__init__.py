
from image_classifier.models.mnist import build_model as build_mnist_model
from enum import Enum
import os

class ModelType(Enum):
    IMAGE_CLASSIFICATION_MNIST = 1
    IMAGE_CLASSIFICATION_CIFAR = 2

def instantiate_model(model_type: ModelType):
    if model_type == ModelType.IMAGE_CLASSIFICATION_MNIST:
        return build_mnist_model()
    elif model_type == ModelType.IMAGE_CLASSIFICATION_MNIST:
        raise ValueError(f'TODO build cifar model: {model_type}')
    else:
        raise ValueError(f'Unknown model type: {model_type}')
