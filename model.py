import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FacialExpressionModel(object):
    emotion_dict = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

    def __init__(self, model_weights_file):
    	self.loaded_model = Sequential()

    	self.loaded_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    	self.loaded_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    	self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
    	self.loaded_model.add(Dropout(0.25))

    	self.loaded_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    	self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
    	self.loaded_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    	self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
    	self.loaded_model.add(Dropout(0.25))

    	self.loaded_model.add(Flatten())
    	self.loaded_model.add(Dense(1024, activation='relu'))
    	self.loaded_model.add(Dropout(0.5))
    	self.loaded_model.add(Dense(7, activation='softmax'))
    	self.loaded_model.load_weights(model_weights_file)
    	self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
    	self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.emotion_dict[int(np.argmax(self.preds))]
