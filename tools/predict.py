
import argparse
import numpy as np
import torch

from image_classifier.data.mnist import MNISTDataset
from image_classifier.models.mnist import load_model, predict_proba

def main():
    parser = argparse.ArgumentParser(
        prog='Predict from a neural net',
        description='Example code for making a prediction from a neural net',)
    
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    dataset = MNISTDataset(img_dir=None, annotations_file=None, load_data=False)
    X = dataset.load_image(args.input)
    X =  torch.from_numpy(X)

    print(f"Loading model... {args.model}")
    model = load_model(args.model)
    print(f"Predicting... {args.input}")
    output = model(X)
    probabilities, indices = torch.max(output, 1)
    index = indices[0].item()
    probability = probabilities[0].item()
    print(f"Predicted {index} with probability {probability}")


if __name__ == "__main__":
    main()
