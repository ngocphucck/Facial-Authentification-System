import cv2
import face_alignment
import numpy as np
from skimage import transform as trans


def alignment(cv_img, dst):
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


def align_image(image_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
     flip_input=False, device='cpu')
    image_path = '../data/demo/detection/1.jpg'
    image = cv2.imread(image_path)

    landmarks = fa.get_landmarks(image)
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

        face = alignment(cv_img, dst)
    
    cv2.imwrite('../data/demo/alignment/' + image_path.split('/')[-1], face)


if __name__ == '__main__':
    align_image('../data/demo/detection/1.jpg')
    pass
