"""Submission Studio 2 - efficientnet_b2 (timm), input 288 px."""
import timm
from torchvision import transforms

NUM_CLASSES = 5
IMG_SIZE = 288
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model():
    """Bangun arsitektur saja - TIDAK mengunduh bobot / memuat model.pth di sini."""
    return timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
