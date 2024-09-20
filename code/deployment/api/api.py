import os
import numpy as np
from pathlib import Path

# ML / DL imports
import torch

# fastapi imports
from fastapi import FastAPI, Request

# custom imports
from code.models.model import MNISTClassifier

# torch device
device = "cuda" if torch.cuda.is_available() else "cpu"

# repo's root directory
root_path = Path(os.getcwd())

# load model
state_dict = torch.load(root_path / "models" / "state_dict.pt", weights_only=False)
model: torch.nn.Module = MNISTClassifier()
model.load_state_dict(state_dict)
model.to(device)
model.freeze()

app = FastAPI(
    title="MNIST Classifier",
    description="""Let's train skills in MLOps!""",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {"message": "omg PMLDL course TA hiiiiiii!"}


@app.post("/predict")
async def predict(request: Request):
    image_bytes: bytes = await request.body()

    # convert image bytes to tensor and put on the same device as model
    image = torch.tensor(np.frombuffer(image_bytes, dtype=np.uint8)).reshape(28, 28)
    image = image.to(device, dtype=torch.float)

    # make prediction
    probs: torch.Tensor = model(image)
    pred = probs.argmax(dim=1)

    # extract values to return
    probs = probs[0].tolist()
    pred = pred.item()

    return {"probs": probs, "pred": pred}
