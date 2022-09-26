import cv2
import numpy as np
import os
import json
import sys
import math
sys.path.append(".")
import image_alignment.face_alignment as face_alignment
import numpy as np
from skimage import transform as trans


def landmarks_alignment(cv_img, dst):
    dst_w = 224
    dst_h = 224

    src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img


def keypoints_alignment(image, eye_coors):
    left_eye, right_eye = eye_coors
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Calculate the angle between the eye points
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    (h, w) = int(math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)), \
             int(math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))

    # image = cv2.circle(image, left_eye, 30, (255, 0, 0), 2)
    # image = cv2.circle(image, right_eye, 30, (255, 0, 0), 2)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    cv2.waitKey(0)

    return rotated_image


def landmarks_align_image(image_path, points=None, demo=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, 
                                      device='cpu')
    image = cv2.imread(image_path)

    landmarks = fa.get_landmarks_from_image(image_path)

    points = landmarks[0]
    p1 = np.mean(points[36:42,:], axis=0)
    p2 = np.mean(points[42:48,:], axis=0)
    p3 = points[33,:]
    p4 = points[48,:]
    p5 = points[54,:]
                            
    if np.mean([p1[1],p2[1]]) < p3[1] \
        and p3[1] < np.mean([p4[1],p5[1]]) \
        and np.min([p4[1], p5[1]]) > np.max([p1[1], p2[1]]) \
        and np.min([p1[1], p2[1]]) < p3[1] \
        and p3[1] < np.max([p4[1], p5[1]]):

        dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
        cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face = landmarks_alignment(cv_img, dst)
    
    cv2.imwrite('data/demo/alignment/' + image_path.split('/')[-1], face)
    if demo:
        cv2.imwrite(image_path, face)


def keypoints_align_image(image_path, points=None, demo=False):
    image = cv2.imread(image_path)
    coors = points

    face = keypoints_alignment(image, (coors[0], coors[1]))
    cv2.imwrite('data/demo/alignment/' + image_path.split('/')[-1], face)

    if demo:
        cv2.imwrite(image_path, face)


def align_image(image_path, points=None, demo=False, t="keypoints"):
    if t == "keypoints":
        keypoints_align_image(image_path, points, demo)
    else:
        landmarks_align_image(image_path, points, demo)


if __name__ == '__main__':
    keypoints_align_image('data/demo/detection/1.jpg')
    pass
