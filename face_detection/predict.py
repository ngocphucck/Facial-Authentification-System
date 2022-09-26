import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import json

import sys
sys.path.append('.')
from face_detection.models import build_ssd, YoloDetector
from face_detection.utils import BaseTransform
import warnings
warnings.filterwarnings("ignore")


# trained_model_path = 'face_detection/weights/ssd300_FACE_18000.pth'
# num_classes = 1 + 1  # +1 background
# net = build_ssd('test', 300, num_classes)  # initialize SSD
# net.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
# net.eval()
net = None
model = YoloDetector(target_size=720, gpu=-1, min_face=90)

def ssd_predict(image_path, save_folder='data/demo/detection', get_ax=False):
    if type(image_path) == str:
        image = Image.open(image_path)    
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, 1)
    else:
        image = image_path
        if type(image) != np.ndarray:
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, 1)
        
    transform = BaseTransform(net.size, (104, 117, 123))

    x = torch.from_numpy(transform(image)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    dets = y.data
    faces = []
    axes = []
    scale = torch.Tensor([
        image.shape[1], image.shape[0],
        image.shape[1], image.shape[0]]
    )

    j = 0
    while dets[0, 1, j, 0] >= 0.4:
        score = dets[0, 1, j, 0]
        box = [score.item()] + (dets[0, 1, j, 1:] * scale).cpu().numpy().tolist()
        j += 1
        cut_image = image[int(box[2]): int(box[4]), int(box[1]): int(box[3]), :]
        cut_image_path = os.path.join(save_folder, str(j) + '.jpg')
        faces.append(cut_image)
        axes.append([int(box[1]), int(box[2]), int(box[3])-int(box[1]), int(box[4])-int(box[2])])
        cv2.imwrite(cut_image_path, cut_image)

    if get_ax:
        return axes
    return faces


def yoloface_predict(image_path, save_folder='data/demo/detection', get_ax=False):
    if type(image_path) == str:
        image = Image.open(image_path)    
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, 1)
    else:
        image = image_path
        if type(image) != np.ndarray:
            image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, 1)

    x = torch.from_numpy(image).permute(2, 0, 1)
    faces, points = model(image)
    
    cut_faces = []
    list_points = []
    j = 0
    axes = []
    for (xmin, ymin, xmax, ymax) in faces[0]:
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
            
        cut_image = image[y: y + h, x: x + w, :]
        cut_faces.append(cut_image)
        axes.append([x, y, w, h])

        for i in range(5):
            points[0][j][i][0] -= x
            points[0][j][i][1] -= y
        list_points.append(points[0][j])
        j += 1
        cv2.imwrite(os.path.join(save_folder, str(j) + '.jpg'), cut_image)
    
    if get_ax:
        return axes
    
    return cut_faces, list_points


def predict(image_path, save_folder='data/demo/detection', get_ax=False, type='yoloface'):
    if type == 'yoloface':
        yoloface_predict(image_path, save_folder, get_ax)
    elif type == 'ssd':
        ssd_predict(image_path, save_folder, get_ax)


if __name__ == '__main__':
    predict('data/demo/original/example.jpg', save_folder='data/demo/detection', type='yoloface')
    pass
