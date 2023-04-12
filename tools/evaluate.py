
import argparse
import os

from image_classifier.data.mnist import MNISTDataset
from image_classifier.models.mnist import load_model
from image_classifier.evaluators.image_classification import evaluate

def main():
    parser = argparse.ArgumentParser(
        prog='Evaluate a neural net',
        description='Example code for evaluating a neural net',)
    
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    img_dir = args.data
    annotations_file = os.path.join(args.data, "test.csv")
    dataset = MNISTDataset(img_dir, annotations_file)

    print(f"Loading model...")
    model = load_model(args.model)
    model.eval()
    print(f"Evaluating model...")
    evaluate(model, dataset, annotations_file, args.output)

if __name__ == "__main__":
    main()
