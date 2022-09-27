import enum
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
import imghdr

fakedata_dir = r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\fake_images\*'

detector = Detection()
cropper = CropImage()
out_w, out_h = 80, 80
scale = [1.0, 2.7, 4.0]


def get_fake_images(dir, type_):
    count = 0
    if type_ == 'live':
        fol = 2
    elif type_ == 'spoof':
        fol = 1
    for folder in glob.glob(dir):
        # For digital images
        digital_dir = os.path.join(dir, folder, str(type_), '*')
        for ind, file in enumerate(glob.glob(digital_dir)):
            if count == 300:
                break
            else:
                if file[-3::] == 'png':
                    img = cv2.imread(file)
                    cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\org_1_80x60\{}\{}.png'.format(int(fol), int(count)), img)
                    for idx, s in enumerate(scale):
                        bbox = detector.get_bbox(img)
                        dst_img = cropper.crop(img, bbox, s, out_w, out_h, crop=True)
                        img_rgb = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
                        if idx == 0:
                            cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\1_80x80\{}\{}.png'.format(int(fol), int(count)), img_rgb)
                        elif idx==1:
                            cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\2.7_80x80\{}\{}.png'.format(int(fol), int(count)), img_rgb)
                        elif idx==2:
                            cv2.imwrite(r'C:\Users\daidv8\Desktop\CV project\Facial-Authentification-System\face_anti_spoofing\Silent-Face-Anti-Spoofing-master\datasets\rgb_image\4_80x80\{}\{}.png'.format(int(fol), int(count)), img_rgb)
                    count+=1
                    print(count)
                

get_fake_images(fakedata_dir, 'live')
get_fake_images(fakedata_dir, 'spoof')




