import os
import io
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

IMAGES_WITH_BOXES = "images_with_boxes"
IMAGES_UPLOADED = "images_uploaded"

#Functions for dir managment
def init_images_dir():
    dir_name = IMAGES_WITH_BOXES

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def init_uploaded_images_dir():
    dir_name = IMAGES_UPLOADED
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

init_images_dir()
init_uploaded_images_dir()

### To run the application:
### uvicorn service:app --reaload

#Create a fastAPI instance
app = FastAPI(title ="ML model with FastAPI")

#class to list available models
class Model(str, Enum):
    yolov4tiny ="yolov4-tiny" #light model
    yolov4 = "yolov4" #full model

#Define behavior of GET method
#Uses the / endpoint
@app.get("/")
def home():
    return "API is working correctly. Go to http://localhost:8000/docs for use."

#Handles the object detection
#at the /predict endpoint.
@app.post("/predict")
def prediction(model: Model, confidence: float =0.5, file: UploadFile = File(...)):
    """
    model:
        model name
    confidence:
        confidence in detection objects
        lower it detects more objects but less accurate.
        higher it detects less objects but more accurate.
    file: image file to predict as a stream of bytes.
    """

    #Talidate input file. Must be an image
    filename = file.filename
    is_image = filename.split(".")[-1] in ("jpg", "jpeg", "png")

    if not is_image:
        raise HTTPException(status_code =415,
            detail = "Unsupported file format. It must be an image with jpg, jpeg or png format.")
    
    #Transform image into cv2 image
    #Read image as a strean if bytes
    image_stream = io.BytesIO(file.file.read())

    #Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()),dtype =np.uint8)

    #Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    #Run object detection on  image
    bbox, label, conf = cv.detect_common_objects(
        image, model = model,confidence = confidence)
    
    #Create image with objects bounding boxes and labels.
    output_image = draw_bbox(image, bbox, label, conf)

    #Save the prediction as an image in the server.
    cv2.imwrite(f'{IMAGES_WITH_BOXES}/{filename}', output_image)

    #Return image to the user in the server.
    #open image in binary mode to streaming it.
    file_image = open(f'{IMAGES_WITH_BOXES}/{filename}', mode='rb')

    #Return the image as a stream passing media type.
    return StreamingResponse(file_image,media_type ="image/jpeg")


