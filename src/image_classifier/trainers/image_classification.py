

from image_classifier.models.mnist import model_path
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import os

def train(
    model: nn.Module, 
    dataloader: DataLoader, 
    epochs: int = 10,
    save_dir: str = "output",
    lr: float = 0.001,
    momentum=0.9,
    save_every: int = 10000,
    print_every: int = 2000,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % save_every == 0:
                full_path = model_path(save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                print(f"Saving model to {full_path}")
                torch.save(model.state_dict(), full_path)

            if i % print_every == 0 and i > 0:
                print(f'[{epoch + 1}, {i:5d}] loss: {running_loss / print_every:.3f}')
                running_loss = 0.0

    print('Finished Training Saving...')
