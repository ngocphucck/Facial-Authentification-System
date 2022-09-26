import cv2
import time 
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

import sys
sys.path.append(".")
from PIL import Image

resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu'
)
load_data = torch.load('deployment/assets/embeddings.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

camera=0
cam = cv2.VideoCapture(camera)

while True:

    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate([prob_list]):
            if prob>0.90:
                emb = resnet(img_cropped_list.unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.90:
                    frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                    
                frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)

    cv2.imshow("IMG", frame)
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
        
cam.release()
cv2.destroyAllWindows()
    