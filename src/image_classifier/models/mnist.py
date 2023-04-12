
import torch
from torch import nn
import os

# defining a CNN architecture 
class MNISTImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X

def model_path(save_dir) -> str:
    return os.path.join(save_dir, 'params.pt')

# creating model
def build_model() -> MNISTImageClassifier:
    model = MNISTImageClassifier()
    return model

def load_model(save_dir: str):
    params_path = model_path(save_dir)
    model = MNISTImageClassifier()
    model.load_state_dict(torch.load(params_path))
    return model

def predict(model: nn.Module, X):
    print(X.shape)
    outputs = model(X[0])
    _probabilities, indices = torch.max(outputs, 1)
    return indices

def predict_proba(model: nn.Module, X):
    print(X.shape)
    outputs = model(X[0])
    probabilities, _indices = torch.max(outputs, 1)
    return probabilities