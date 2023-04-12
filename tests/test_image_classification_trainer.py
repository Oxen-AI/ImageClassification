
from image_classifier.data.mnist import MNISTDataset
from image_classifier.models import ModelType, instantiate_model
from image_classifier.trainers.image_classification import train
from torch.utils.data import DataLoader

import os
import shutil

def test_image_classification_trainer():
    data_dir = 'tests/data/mnist'
    img_dir = 'tests/data/mnist'
    annotations_file = os.path.join(img_dir, "train.csv")
    dataset = MNISTDataset(img_dir, annotations_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    save_dir = os.path.join(data_dir, 'models')

    # Clean Up
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    model = instantiate_model(ModelType.IMAGE_CLASSIFICATION_MNIST)

    train(model, dataloader, epochs=10, save_dir=save_dir)
    assert os.path.exists(os.path.join(save_dir, 'params.pt'))
    # Clean Up
    shutil.rmtree(save_dir)