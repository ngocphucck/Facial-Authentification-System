Facial Authentification System
=====

# Introduction

This project aims to build a facial authentification app which includes face detection, face recognition, face anti-spoofing attacks and sentiment analysis to contribute to better authenticated system. The flow of this system is described in the following figure:

<img width="667" alt="flow" src="https://user-images.githubusercontent.com/53470099/193184374-bf18a188-7267-473c-9198-e9ab2fc9c79b.png">

# Usage
```bash
git clone https://github.com/ngocphucck/Facial-Authentification-System.git
cd Facial-Authentification-System
docker build -t "app" .
docker run app
```
# Demo
![demo](https://user-images.githubusercontent.com/53470099/193184281-7ea17e9b-809a-4684-845a-4a92ffab7b43.gif)

# References
Thanks to some awesome works:

- [mtcnn](https://github.com/timesler/facenet-pytorch)
- [yolofacev5](https://github.com/elyha7/yoloface])
