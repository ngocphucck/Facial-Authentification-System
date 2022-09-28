from distutils.log import debug
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import json
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import sys
sys.path.append(".")
from face_detection.predict import predict as detect
from face_detection.predict import ssd_predict, yoloface_predict
from image_enhacement.srgan.tools.predict import predict as enhance
from image_alignment.alignment import align_image
from face_recognition.facenet.test import *
from face_anti_spoofing.FAS.infer import check_fake
import torch
detector = cv2.CascadeClassifier('deployment/haarcascade_frontalface_default.xml')


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def detect_faces(image_path):
    # image = load_image(image_path)
    # image = np.array(image.convert('RGB'))
    # image = cv2.cvtColor(image, 1)
    faces = yoloface_predict(image_path)
    # path_to_face = f"data/demo/detection/{name}"
    return faces


def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Edges
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    # Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


def main(img, get_axes=False):
    '''main function for recognizing people
    '''
    if get_axes:
        name, faces = recognize(img, get_axes=True)
        return json.load(open(f"deployment/assets/info/{name}.json",'r')), faces[0]
    else:
        name = recognize(img)
        return json.load(open(f"deployment/assets/info/{name}.json",'r'))


def add_embedding(img_path, user_code):
    ########################################################
    new_name = str(user_code)
    img = np.array(Image.open(img_path))
    img_cropped_list = mtcnn(img, return_prob=False) 
    new_emb = resnet(img_cropped_list.unsqueeze(0)).detach() 
    data = [embedding_list.append(new_emb), name_list.append(new_name)] 
    torch.save(data, 'deployment/assets/embeddings.pt') # saving .pt file


def app():
    """Face Recognition App"""

    st.title("Face Recognition App")
    st.text("Build with Streamlit & Deep learning algorithms")

    activities = ["About", "Upload", "Recognition", "Realtime Webcam Recognition"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'About':
        st.subheader("Face Authentication App")
        st.markdown(
            "Built with Streamlit by [Manh Pham](https://github.com/manhph2211). The web appplication allows to add user to database and verfify them later. Also, we provide solutions for face anti-spoofing and face sentiment analysis ...")
        st.subheader("Team members:")
        members = ''' 
            Pham Hung Manh\n
            Doan Ngoc Phu\n
            Do Van Dai\n
            Ha Bao Anh\n
            Nguyen Xuan Hoang\n'''
        st.markdown(members)
        st.success("Max Ph")

    elif choice == 'Upload':
        st.subheader("Add your face to database")
        user_name = st.text_input("Enter your name:")
        user_dob = st.text_input("Enter your date of birth (dd/mm/yyyy):")
        user_code = st.text_input("Enter your code:")
        image = st.camera_input("Take a picture")
        if image is not None:
            image = Image.open(image)
            image, points = detect_faces(image)
            image, points = image[0], points[0]
            user_counts = len(next(os.walk('deployment/assets/target_imgs'))[1])
            print(f"There are {user_counts} users so far!!!")
            user_folder = os.path.join("deployment/assets/target_imgs",user_code)
            if not os.path.isdir(user_folder):
                os.mkdir(user_folder)
            new_upload_path = os.path.join(user_folder, time.strftime("%Y%m%d%H%M%S.jpg"))
            cv2.imwrite(new_upload_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            # enhance(new_upload_path)
            # align_image(new_upload_path, points, demo=True)
            user_info_path = os.path.join(user_folder.replace(f"target_imgs\{user_code}", "info"), f"{user_code}.json")
            print(user_info_path)
            user_info = {
                "name": user_name,
                "user_code": user_code,
                "user_dob": user_dob,
                "img_num": len([name for name in os.listdir(f'deployment/assets/target_imgs/{user_code}') if name.endswith("jpg")])
            }
            with open(user_info_path,'w') as f:
                json.dump(user_info, f, indent=2)
            
            text, result = check_fake(new_upload_path)
            if result:   
                add_embedding(new_upload_path, user_code)
                st.warning(text)
                st.success(f"Upload : Successfully Saved Embedding!")

            else:
                # add_embedding(new_upload_path, user_code)
                st.warning(text)

    if choice == 'Recognition':
        st.subheader("Face Recognition")
        image = st.camera_input("Take a picture")
        if image is not None:
            image = Image.open(image)

            enhance_type = st.sidebar.radio(
                "Augmentation", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
                
            if enhance_type == 'Gray-Scale':
                new_img = np.array(image.convert('RGB'))
                img = cv2.cvtColor(new_img, 1)
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(result)

            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(image)
                result = enhancer.enhance(c_rate)
                st.image(result)

            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(image)
                result = enhancer.enhance(c_rate)
                st.image(result)

            elif enhance_type == 'Blurring':
                new_img = np.array(image.convert('RGB'))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                img = cv2.cvtColor(new_img, 1)
                result = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(result)

            else:
                # st.image(image, width=300)
                result = image

        if st.button("Process"):
            with st.spinner(text="🤖 Recognizing..."):
                data = main(result)
                st.write(data)
                st.balloons()

    elif choice == "Realtime Webcam Recognition":

        st.warning("NOTE : In order to use this mode, you need to give webcam access.")

        spinner_message = "🤖 Wait a sec, getting some things done..."
        minimum_neighbors = 4
        # Minimum possible object size
        min_object_size = (50, 50)
        minimum_neighbors = st.slider("Mininum Neighbors", min_value=0, max_value=10,
                                    help="Parameter specifying how many neighbors each candidate "
                                        "rectangle should have to retain it. This parameter will affect "
                                        "the quality of the detected faces. Higher value results in less "
                                        "detections but with higher quality.",
                                    value=minimum_neighbors)

        # slider for choosing parameter values

        min_size = st.slider(f"Mininum Object Size, Eg-{min_object_size} pixels ", min_value=3, max_value=500,
                            help="Minimum possible object size. Objects smaller than that are ignored.",
                            value=70)

        min_object_size = (min_size, min_size)
        with st.spinner(spinner_message):

            class VideoProcessor:

                def recv(self, frame):
                    # convert to numpy array
                    frame = frame.to_ndarray(format="bgr24")
                    frame = cv2.cvtColor(frame, 1)

                    # detect faces
                    # cut_faces, list_points, axes = yoloface_predict(frame, get_ax=True)
                    # faces = ssd_predict(frame, get_ax=True)
                    faces = detector.detectMultiScale(frame, 1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

                    # draw bounding boxes
                    # for i, (x, y, w, h) in enumerate(faces):
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        # face = Image.fromarray(cut_faces[i])
                        # info = main(face)
                    info, faces = main(frame, get_axes=True)
                    cv2.rectangle(frame, (int(faces[0]), int(faces[1])), (int(faces[2]), int(faces[3])), (0, 255, 0), 3)
                    frame = cv2.putText(frame, info['name'], (int(faces[0]),int(faces[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                    frame = av.VideoFrame.from_ndarray(frame, format="bgr24")

                    return frame

            webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                            rtc_configuration=RTCConfiguration(
                                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))


if __name__ == '__main__':
    app()