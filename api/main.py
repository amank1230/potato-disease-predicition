from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import gdown

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI()

# Allow frontend requests
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Download model from Google Drive if not already present
# -----------------------------
MODEL_DIR = "save_models/1"
os.makedirs(MODEL_DIR, exist_ok=True)

# Replace FILE_ID with the actual Google Drive file ID for your model
MODEL_FILE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.h5")

if not os.path.exists(MODEL_FILE_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_FILE_URL, MODEL_FILE_PATH, quiet=False)
else:
    print("Model already exists, skipping download.")

# -----------------------------
# Load your trained model
# -----------------------------
MODEL = tf.keras.models.load_model(MODEL_FILE_PATH)

# Use exact class names from training
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# -----------------------------
# Test endpoint
# -----------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive 🚀"} 

# -----------------------------
# Image preprocessing function
# -----------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # match model input size
    image = np.array(image)
    return image

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence,
        "confidence_percent": f"{confidence*100:.2f}%"
    }

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
