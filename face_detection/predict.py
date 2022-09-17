import os
import time
import pdb
from flask import Flask, render_template, request, Response
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import jsonify
import sys
import torch
# import cudnn
from torch.autograd import Variable
from ssd import build_ssd
from utils import BaseTransform
import threading

os.environ["CUDA_VISIBLE_DEVICES"]=""

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


detections = []
frame = None


class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cuda = False
        if self.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        trained_model_path = 'weights/ssd300_FACE_18000.pth'
        num_classes = 1 + 1  # +1 background
        self.net = build_ssd('test', 300, num_classes)  # initialize SSD
        if not self.cuda:
            self.net.load_state_dict(torch.load(
                trained_model_path,
                map_location=lambda storage,
                loc: storage)
            )
        else:
            self.net.load_state_dict(torch.load(trained_model_path))
        self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()
            # import cudnn
            # cudnn.benchmark = True
        self.transform = BaseTransform(self.net.size, (104, 117, 123))

    def run(self):
        global detections, frame
        while True:
            if frame is None:
                continue
            # print(frame.shape)
            detections = self.predict(frame)

    def predict(self, frame):
        #print('Predicting..')
        start = time.time()
        # img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        x = torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if self.cuda:
            x = x.cuda()
        y = self.net(x)
        dets = y.data
        bboxes = []
        scale = torch.Tensor([
            frame.shape[1], frame.shape[0],
            frame.shape[1], frame.shape[0]]
        )
        j = 0
        while dets[0, 1, j, 0] >= 0.4:
            score = dets[0, 1, j, 0]
            box = [score.item()] + (dets[0, 1, j, 1:] * scale).cpu().numpy().tolist()
            bboxes.append(box)  # XMin, YMin, XMax, YMax
            j += 1
        print('Time taken: ', time.time() - start)
        return bboxes



Modelthread = MyThread()
Modelthread.start()
print(detections)


@app.route('/postImage', methods=['GET', 'POST'])
def postImage():
    global frame
    img64 = request.form['image'].replace("data:image/png;base64,", "")
    image = Image.open(BytesIO(base64.b64decode(img64))).convert("RGB")
    frame = np.asarray(image, dtype=np.float64)
    # detections = Modelthread.run()
    # detections = [[0.80586177110672,111.9474105834961,79.74383544921875,330.58477783203125,285.7025146484375], [0.80586177110672,111.9474105834961,89.74383544921875,350.58477783203125,295.7025146484375]]

    # return jsonify(bboxes)
    return jsonify(detections)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, threaded=True)
