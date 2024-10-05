from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from io import BytesIO  # For handling the in-memory file
from PIL import Image  # To handle image files without saving

app = FastAPI()

# Set up CORS middleware
origins = [
    "http://localhost:5173",  # The origin of the frontend server
    "http://localhost:3000",  # The origin of the frontend server
    "https://yourproductionwebsite.com"  # If you have a production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained Keras model
model = load_model('skin_modal.keras')

CLASSES = ['Acne', 'Eczema', 'Healthy', 'Vitiligo']

def transform_image(image):
    image_size = (224, 224)
    image = image.resize(image_size)  # Resize using PIL
    image = np.array(image).astype(np.float32) / 255.0  # Convert to NumPy array and normalize
    if image.shape[-1] == 4:  # If the image has an alpha channel, remove it
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
async def home():
    return {"Message": "SERVER IS RUNNING FINE"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Convert uploaded file into an image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")  # Convert to RGB
        
        # Transform image
        transformed_image = transform_image(image)

        # Make prediction
        prediction = model.predict(transformed_image)
        class_index = np.argmax(prediction)
        class_name = CLASSES[class_index]

        # Create a response with class name and probabilities
        probabilities = [{"name": name, "probability": round(float(val), 4)} for name, val in zip(CLASSES, prediction[0])]

        return {"prediction": class_name, "probabilities": probabilities}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}

@app.get("/test")
async def test():
    return {"message": "TEST IS RUNNING FINE"}