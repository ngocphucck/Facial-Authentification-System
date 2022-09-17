import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = None
    return img, faces


def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = None
    return img, eyes


def detect_smiles(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect Smiles
    smiles = None
    return img


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
            image = Image.open(image)
            user_counts = len(next(os.walk('deployment/assets/target_imgs'))[1])
            print(f"There are {user_counts} users so far!!!")
            user_folder = os.path.join("deployment/assets/target_imgs",user_code)
            if not os.path.isdir(user_folder):
                os.mkdir(user_folder)
            new_upload_path = os.path.join(user_folder, time.strftime("%Y%m%d%H%M%S.jpg"))
            cv2.imwrite(new_upload_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))



    if choice == 'Verify':
        st.subheader("Face Verification")

        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

            enhance_type = st.sidebar.radio(
                "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == 'Gray-Scale':
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # st.write(new_img)
                st.image(gray)
            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            elif enhance_type == 'Blurring':
                new_img = np.array(our_image.convert('RGB'))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
                img = cv2.cvtColor(new_img, 1)
                blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(blur_img)
            elif enhance_type == 'Original':

                st.image(our_image, width=300)
            else:
                st.image(our_image, width=300)

        # Face Detection
        task = ["Faces", "Smiles",
                "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):

            if feature_choice == 'Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)

                st.success("Found {} faces".format(len(result_faces)))
            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img)

            elif feature_choice == 'Eyes':
                result_img, result_eyes = detect_eyes(our_image)
                st.success("Found {} Eyes".format(len(result_eyes)))
                st.image(result_img)

            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img)

            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny)

            elif feature_choice == 'Eyes and Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)
                result_img = detect_eyes(our_image)
                st.image(result_img)
                st.success("Found {} faces and {} eyes".format(
                    len(result_faces), len(result_eyes)))



if __name__ == '__main__':
    main()