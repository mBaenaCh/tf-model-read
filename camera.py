import cv2
from model import FlowerRecognitionModel
import numpy as np

model = FacialRecognitionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        inp = cv2.resize(frame, (150,150))
        pred = model.predict_img(inp)
        print(pred)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()