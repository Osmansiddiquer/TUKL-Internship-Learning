import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras import models
from fastapi import FastAPI, File, UploadFile

PATH_TO_MODEL = "./model/saved-models/model-v1.keras"
MODEL = models.load_model(PATH_TO_MODEL)

MODEL.summary()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
@app.post("/predict")
async def precict(file: UploadFile):
    image = np.array(Image.open(BytesIO(await file.read())))
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence * 100)
    }