# Real Time Emotion Prediction
A deep CNN to classify the facial emotion into 7 categories. The model is trained on the **FER-2013** dataset which was a part of kaggle FER-2013 challenge. This dataset consists of 35887 grayscale, 48x48 sized face images with **7 categories** as follows : angry, disgusted, fearful, happy, neutral, sad, surprised. We have used haar cascade method for face detection. We create softmax scores for all 7 categories and emotion with maximum score is displayed.

## Dependencies
To install all required dependencies, run `pip install -r requirements.txt`

## Usage
* Clone the repository and enter into the directory

```bash
git clone https://github.com/janvishah21/emotion_prediction.git
cd emotion_prediction
```

* To train the model, we have converted actual csv file into a dataset of images, you can download dataset from [here](https://drive.google.com/file/d/1rkC29dRCaq8TZBh0ZRFANHwW3-PeIsrS/view?usp=sharing) and unzip it to create directory `data` right inside main repository. For training, run

```bash
python train_model.py
```

This will train a new model and its weights will be stored in `model.h5` file. If you want to skip this step(do not want to train your model), you can use our pre-trained `model.h5` model further.

* When trained model is ready, run

```bash
python main.py
```
Now, go to http://127.0.0.1:5000/ in your browser...