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
# Download and extract model from Google Drive if not already present
# -----------------------------
MODEL_DIR = "saved_models/1"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive zip file link (replace YOUR_FILE_ID with your actual file ID)
# Your link: https://drive.google.com/file/d/1DLga0IVMdVXd0GwSveTIUaoaQKb6YeCC/view?usp=drive_link
FILE_ID = "1DLga0IVMdVXd0GwSveTIUaoaQKb6YeCC"
ZIP_PATH = "saved_model.zip"

if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print("Downloading model zip from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    print("Extracting model zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(ZIP_PATH)  # Clean up zip after extraction
else:
    print("Model already exists, skipping download.")

# -----------------------------
# Load your trained model
# -----------------------------
MODEL = tf.keras.models.load_model(MODEL_DIR)

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
