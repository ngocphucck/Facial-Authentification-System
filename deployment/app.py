from distutils.log import debug
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time
import json
import sys
sys.path.append(".")
from face_detection.predict import predict


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def detect_faces(image_path):
    # image = load_image(image_path)
    # image = np.array(image.convert('RGB'))
    # image = cv2.cvtColor(image, 1)
    face = predict(image_path)
    # path_to_face = f"data/demo/detection/{name}"
    return face[0]


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


def verify():
    '''dump example
    '''
    return json.load(open("deployment/assets/target_imgs/20180134/20180134.json",'r'))


def main():
    """Face Verification App"""

    st.title("Face Verification App")
    st.text("Build with Streamlit & Deep learning algorithms")

    activities = ["Upload", "Verify", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'About':
        st.subheader("Face Authentication App")
        st.markdown(
            "Built with Streamlit by [Max](https://github.com/manhph2211). The web appplication allows to add user to database and verfify them later. Also, we provide solutions for face anti-spoofing and face sentiment analysis ...")
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
            # image = Image.open(image)
            image = detect_faces(image)
            user_counts = len(next(os.walk('deployment/assets/target_imgs'))[1])
            print(f"There are {user_counts} users so far!!!")
            user_folder = os.path.join("deployment/assets/target_imgs",user_code)
            if not os.path.isdir(user_folder):
                os.mkdir(user_folder)
            new_upload_path = os.path.join(user_folder, time.strftime("%Y%m%d%H%M%S.jpg"))
            cv2.imwrite(new_upload_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

            user_info_path = os.path.join(user_folder, f"{user_code}.json")
            user_info = {
                "name": user_name,
                "user_code": user_code,
                "user_dob": user_dob,
                "img_num": len([name for name in os.listdir(f'deployment/assets/target_imgs/{user_code}') if name.endswith("jpg")])
            }
            with open(user_info_path,'w') as f:
                json.dump(user_info, f, indent=2)

    if choice == 'Verify':
        st.subheader("Face Verification")
        image = st.camera_input("Take a picture")
        if image is not None:
            image = Image.open(image)

            enhance_type = st.sidebar.radio(
                "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
                
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
            with st.spinner(text="This may take a moment..."):
                data = verify()
                st.write(data)


if __name__ == '__main__':
    main()