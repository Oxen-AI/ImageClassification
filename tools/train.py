
import argparse
import os
from torch.utils.data import DataLoader

from image_classifier.data.mnist import MNISTDataset
from image_classifier.models import ModelType
from image_classifier.trainers.image_classification import train
from image_classifier.models import instantiate_model

def main():
    parser = argparse.ArgumentParser(
        prog='Train a neural net',
        description='Example code for training a neural net',)
    
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-e', '--epochs', default=50)
    parser.add_argument('-b', '--batch_size', default=4)
    parser.add_argument('-p', '--print_every', default=2000)
    parser.add_argument('-s', '--save_every', default=10000)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}")
    img_dir = args.data
    annotations_file = os.path.join(args.data, "train.csv")
    dataset = MNISTDataset(img_dir, annotations_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"Instantiating model...")
    model = instantiate_model(ModelType.IMAGE_CLASSIFICATION_MNIST)

    save_dir = os.path.join(args.output, 'models')
    train(
        model,
        dataloader,
        epochs=args.epochs,
        save_dir=save_dir,
        print_every=args.print_every,
        save_every=args.save_every
    )

    print(f"Done training, saved to {save_dir}")

if __name__ == "__main__":
    main()
