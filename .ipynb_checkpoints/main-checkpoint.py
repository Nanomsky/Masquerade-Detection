from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tempfile
import os

app = FastAPI()

#MODEL = tf.keras.models.load_model("/Users/ositanwegbu/Documents/GitHub/Masquerade_detection/Model/Model_1.h5")
MODEL = tf.keras.models.load_model("./Model/Model_1.h5")
CLASS_NAMES = ['No_masquerade', 'masquerade']

@app.get("/ping")
async def ping():

    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict/")
async def predict(
    file: UploadFile
):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())

    try:
        # Load and preprocess the image
        img = image.load_img(temp_file.name, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension if needed
        predictions = MODEL.predict(img_array)
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class':predicted_class,
        'confidence': f'{round(float(confidence)*100,2)}%'
    }
















'''
@app.post("/predict")
async def predict(

    file: UploadFile = File(...) 
    
    ):
    
    

    # Load and preprocess the image
    img = image.load_img(file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension if needed

    # Now, you can pass `img_array` to your model's predict method
    predictions = MODEL.predict(img_array)
    
    
    
    
    
    #image = read_file_as_image(await file.read())
    #image_resize = image.resize((128,128))
    #img_batch = np.expand_dims(image_resize,0) #converts [256,256,3] to [[256,256,3]] since our model expects a batch instead of a single image
    #img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    #predictions = MODEL.predict(img_tensor)    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class':predicted_class,
        'confidence': float(confidence)
    }'''
if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)
    
    
    