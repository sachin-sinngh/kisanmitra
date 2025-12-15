import json
import pickle
import cv2
import numpy as np
import os
from flask import Flask, request
import time
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

# load model and label
model = pickle.load(open('model.pkl', 'rb'))
image_labels = pickle.load(open('label.pkl', 'rb'))

IMAGE_SIZE = tuple((256, 256))

def toArray(filepath):
  try:
    image = cv2.imread(filepath)
    if image is not None:
      image = cv2.resize(image, IMAGE_SIZE)   
      return img_to_array(image)
    else:
      return np.array([])
  except Exception as e:
    print(f"Error : {e}")
    return None

def predict(image_path):
  image_array = toArray(image_path)
  np_image = np.array(image_array, dtype=np.float16) / 225.0
  np_image = np.expand_dims(np_image,0)
  result = model.predict(np_image)
  classes_x=np.argmax(result,axis=1)
  ans = image_labels.classes_[classes_x][0]
  print(ans)
  return ans

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
  return "Alive"

@app.route('/predict', methods=['POST'])
def index():
  result = "Sorry, Something went wrong!"
  print(request)
  try:
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('./uploads', filename)
    file.save(filepath)
    result = predict(filepath)
    print(result)
    os.remove(filepath)
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

  return result

if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
