import glob
import cv2
import numpy as np
import argparse
from src.anti_spoof_predict import Detection
import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image 
from src.generate_patches import CropImage


img_folder_dir = r"C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\data512x512\*"
count=0
detector = Detection()
cropper = CropImage()
out_w, out_h = 80, 80

for ind, org_img_dir in enumerate(glob.glob(img_folder_dir)):
    if ind == 300:
        break
    else:
        org_img = cv2.imread(org_img_dir)
        cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\org_1_80x60\0\{}.jpg'.format(ind), org_img)
        scale = [1.0, 2.7, 4.0]
        for idx, s in enumerate(scale):
            bbox = detector.get_bbox(org_img)
            dst_img = cropper.crop(org_img, bbox, s, out_w, out_h, crop=True)
            img_rgb = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img_rgb)
            temp = 0
            if idx == 0:
                cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\1_80x80\0\{}.jpg'.format(ind), img_rgb)
            elif idx==1:
                cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\2.7_80x80\0\{}.jpg'.format(ind), img_rgb)
            elif idx==2:
                cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\4_80x80\0\{}.jpg'.format(ind), img_rgb)
        count+=1