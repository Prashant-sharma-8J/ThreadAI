import os
import io
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "resnet50_best.pth")
STYLE_MODEL_PATH = os.path.join(BASE_DIR, "resnet50_style_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

category_checkpoint = torch.load(CATEGORY_MODEL_PATH, map_location=DEVICE)

IDX_TO_CLASS = category_checkpoint["idx_to_class"]
NUM_CLASSES = len(IDX_TO_CLASS)

category_model = models.resnet50(weights=None)
category_model.fc = nn.Linear(category_model.fc.in_features, NUM_CLASSES)
category_model.load_state_dict(category_checkpoint["model_state"])
category_model.to(DEVICE)
category_model.eval()

style_checkpoint = torch.load(STYLE_MODEL_PATH, map_location=DEVICE)

STYLE_CLASSES = style_checkpoint["idx_to_class"]

style_model = models.resnet50(weights=None)
style_model.fc = nn.Linear(
    style_model.fc.in_features,
    len(STYLE_CLASSES)
)

style_model.load_state_dict(style_checkpoint["model_state"])
style_model.to(DEVICE)
style_model.eval()


app = FastAPI(title="ThreadAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/api/threadai/health")
def health():
    return {"status": "ThreadAI backend running"}

def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)


def prettify_label(label: str) -> str:
    return " ".join(word.capitalize() for word in label.split("_"))

@app.post("/api/threadai/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img_tensor = preprocess_image(img_bytes)

    with torch.no_grad():
        # Category model
        cat_outputs = category_model(img_tensor)
        cat_probs = torch.softmax(cat_outputs, dim=1)
        cat_conf, cat_pred = torch.max(cat_probs, 1)

        # Style model
        style_outputs = style_model(img_tensor)
        style_probs = torch.softmax(style_outputs, dim=1)
        style_conf, style_pred = torch.max(style_probs, 1)

    # Format labels
    raw_category = IDX_TO_CLASS[cat_pred.item()].replace("_merged", "")
    category_label = prettify_label(raw_category)

    raw_style = STYLE_CLASSES[style_pred.item()]
    style_label = prettify_label(raw_style)

    return {
        "category": {
            "label": style_label,
            "confidence": round(style_conf.item(), 4)
        },
        "style": {
            "label": category_label,
            "confidence": round(cat_conf.item(), 4)
        }
    }