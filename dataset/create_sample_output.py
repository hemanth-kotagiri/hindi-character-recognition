import random
import os
import json
from tensorflow import keras
import numpy as np
import cv2 as cv
# from utils.io import write_json


def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)


def read_json(filename):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    return data


def generate_sample_file(filename):
    # loading the model
    model = keras.models.load_model("../model_save")
    testing_dir = "./data/test"
    res = {}
    for image_name in os.listdir(testing_dir):
        image_path = os.path.join(testing_dir, image_name)
        image = cv.imread(image_path, 0) / 255.0
        pred = model.predict(np.array(image).reshape(1, 64, 64, -1))
        prediction = 1 if pred > 0.5 else 0
        res[image_name] = prediction

    write_json(filename, res)


if __name__ == '__main__':
    generate_sample_file('./sample_result1.json')
