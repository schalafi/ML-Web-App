# ML-Web-App
`python 3.10`

Object detection web application with python and FastAPI.
Use as an independent service/app to connect with other services.

#### Dependencies
Install dependencies with: `pip install -r requirements.txt`

#### Usage

Use as a web app, with basic interface.


To run locally use with python `uvicorn service:app`

Go to `http://localhost:8000/docs` to send images to the endpoint and run the detection.

Upload an image

![input image](https://user-images.githubusercontent.com/118856089/210655415-8bb921e2-df3f-45d3-be0b-00b100b27272.jpg)

Output an image with bounding boxes and confidence for the detected objects.
You can control the confidence thw model will use to detect objects.
![output image](https://user-images.githubusercontent.com/118856089/210655451-11d8e9bc-9254-4bb3-97d8-8d9174f0496a.jpg)
