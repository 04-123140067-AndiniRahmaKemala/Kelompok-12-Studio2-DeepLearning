
import torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision import transforms

CLASS_NAMES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
]

def get_model():
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        len(CLASS_NAMES)
    )
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
