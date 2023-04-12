
from image_classifier.data.mnist import MNISTDataset
from torch.utils.data import default_collate
import os

def test_mnist_dataloader():
    img_dir = 'tests/data/mnist'
    annotations_file = os.path.join(img_dir, "train.csv")
    dataset = MNISTDataset(img_dir, annotations_file)
    (x_train, y_train) = default_collate(dataset)

    assert x_train.shape == (20, 1, 28, 28)
    assert y_train.shape == (20,)
