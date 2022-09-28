import os
import cv2
import numpy as np
import warnings
import time

from face_anti_spoofing.FAS.src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        print("Current image's size is ", image.shape)
        return False
    else:
        return True


def check_fake(image_path, model_dir="face_anti_spoofing/FAS/resources/anti_spoof_models", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    # Resize the image to height/width = 4/3
    if height/width != 4/3:
        remain = height%4
        new_height = height - remain
        new_width = new_height/4*3
        dim = (int(new_width), int(new_height))
        # resize image
        img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    result = check_image(img)
    if result is False:
        return
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        dim = (80, 80)
        # resize image
        img_ = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        start = time.time()
        prediction += model_test.predict(img_, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if value <0.65:
        if label == 0:
            label = 1
        elif label == 1:
            label = 0
    if label == 1:
        print("Image is Real Face. Score: {:.2f}.".format(value))
        result_text = "RealFace Score: {:.2f}".format(value)
    else:
        print("Image is Fake Face. Score: {:.2f}.".format(value))
        result_text = "FakeFace Score: {:.2f}".format(value)
    print("Prediction cost {:.2f} s".format(test_speed))
    return result_text, label