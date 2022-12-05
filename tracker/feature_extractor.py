import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = models.resnet34(pretrained=True)
transform = transforms.ToTensor()

# remove last fully-connected layer
new_classifier = nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.classifier = new_classifier
feature_extractor.to(device)
feature_extractor.eval()


def extract_features(img: np.ndarray, bbox: list[int]):
    if img is None:
        return np.zeros((1, 1000))
    img = cv2.resize(img, (800, 1000))
    img = crop_img(img, bbox)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    tensor = transform(img)
    tensor = tensor.to(device)
    tensor = tensor.unsqueeze(0)
    features = feature_extractor.forward(tensor)
    features = features.to(torch.device("cpu"))
    features = features.detach().numpy()
    return features


def crop_img(img: np.ndarray, bbox: list[int]):
    x_min = bbox[0] if bbox[0] > 0 else 0
    y_min = bbox[1] if bbox[1] > 0 else 0
    x_max = bbox[2]
    y_max = bbox[3]
    return img[y_min:y_max, x_min:x_max]
