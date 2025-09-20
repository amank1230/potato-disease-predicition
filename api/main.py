from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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
# Load your trained model
# -----------------------------
MODEL = tf.keras.models.load_model("../saved_models/1")

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
    """
    Convert uploaded file bytes to a numpy array
    Matches training input size and channel order
    ⚠️ Do NOT divide by 255.0 here if your model already has Rescaling
    """
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # match model input size
    image = np.array(image)  # keep original scale; model handles rescaling
    return image

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # add batch dimension

    # Predict
    predictions = MODEL.predict(img_batch, verbose=0)

    # Get predicted class and confidence
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
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
