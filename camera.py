import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.h5")
font = cv2.FONT_HERSHEY_DUPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
        	cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        	roi_gray = gray[y:y + h, x:x + w]
        	cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        	prediction = model.predict_emotion(cropped_img)
	        cv2.putText(frame, prediction, (x+20, y-60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', cv2.resize(frame,(800,480), interpolation = cv2.INTER_CUBIC))
        return jpeg.tobytes()
