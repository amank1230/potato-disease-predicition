from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import gdown
import zipfile

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

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
# Model download and extraction
# -----------------------------
MODEL_DIR = "saved_models/1"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive ZIP link (replace with your actual link)
MODEL_ZIP_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, "saved_model.zip")

if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
    print("Downloading model ZIP from Google Drive...")
    gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)

    print("Extracting model ZIP...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
else:
    print("Model already exists, skipping download.")

# -----------------------------
# Load TensorFlow SavedModel
# -----------------------------
MODEL = tf.keras.models.load_model(MODEL_DIR)

# Exact class names
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# -----------------------------
# Test endpoint
# -----------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive 🚀"}

# -----------------------------
# Image preprocessing
# -----------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

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
