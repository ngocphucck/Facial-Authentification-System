import os
from turtle import width
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        print("Current image's size is ", image.shape)
        return False
    else:
        return True

def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    # image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    height, width, channel = image.shape
    print(image.shape)
    # Resize the image to height/width = 4/3
    if height/width != 4/3:
        remain = height%4
        new_height = height - remain
        new_width = new_height/4*3
        dim = (int(new_width), int(new_height))
        # resize image
        img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # print cropped image shape
    print(img.shape)

    result = check_image(img)
    if result is False:
        return
    # image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        # param = {
        #     "org_img": image,
        #     "bbox": image_bbox,
        #     "scale": scale,
        #     "out_w": w_input,
        #     "out_h": h_input,
        #     "crop": True,
        # }
        # if scale is None:
        #     param["crop"] = False
        # img = image_cropper.crop(**param)
        dim = (80, 80)
        # resize image
        img_ = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        start = time.time()
        prediction += model_test.predict(img_, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    print(img_.shape)
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        if value < 0.6:
            result_text = "Might be a fake face. Change your position for another authentification attempt"
            print(result_text)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)

    else:
        if value < 0.6:
            result_text = "Might be a fake face. However, you have to change your position for another authentification attempt"
            print(result_text)
            color = (0, 0, 255)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)




test(image_name='phureal.jpg', model_dir="./resources/anti_spoof_models", device_id=0)
