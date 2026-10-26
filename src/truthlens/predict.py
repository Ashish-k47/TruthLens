
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "truthlens_mobilenetv3_best.pth"
MAPPING_PATH = ROOT_DIR / "class_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(MAPPING_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


def predict_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_idx = int(probabilities.argmax().item())

    return {
        "prediction": idx_to_class[predicted_idx],
        "confidence": round(float(probabilities[predicted_idx]) * 100, 2),
        "probabilities": {
            idx_to_class[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(probabilities))
        }
    }
