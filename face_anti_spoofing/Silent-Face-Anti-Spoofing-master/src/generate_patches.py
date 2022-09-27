"""
Create patch from original input image by using bbox coordinate
"""

import cv2
import numpy as np
import argparse
from src.anti_spoof_predict import Detection
import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image 

class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img

org_img_dir = r"C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\images\dai1.jpg"
org_img = cv2.imread(org_img_dir)

scale = [1.0, 2.7, 4.0]
out_w, out_h = 80, 80
detector = Detection()
bbox = detector.get_bbox(org_img)
cropper = CropImage()

for idx, s in enumerate(scale):
    dst_img = cropper.crop(org_img, bbox, s, out_w, out_h, crop=True)
    img_rgb = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    temp = 0
    if idx == 0:
        direct = r"C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\1_80x80\0"
        filename = os.path.join("{}.jpg".format(temp))
        img_name = os.path.join(direct, filename)
        img.save(img_name)
    elif idx==1:
        direct = r"C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\2.7_80x80\0"
        filename = os.path.join("{}.jpg".format(temp))
        img_name = os.path.join(direct, filename)
        img.save(img_name)
    elif idx==2:
        direct = r"C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\4_80x80\0"
        filename = os.path.join("{}.jpg".format(temp))
        img_name = os.path.join(direct, filename)
        img.save(img_name)