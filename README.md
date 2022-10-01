Facial Authentification System
=====

# Introduction

This project aims to build a facial authentification app which includes face detection, face recognition, face anti-spoofing attacks and sentiment analysis to contribute to better authenticated system. The flow of this system is described in the following figure:

<img width="667" alt="flow" src="https://user-images.githubusercontent.com/53470099/193404457-f3f0b163-5178-489c-8357-b838fa7dc9f4.png">

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

- [MTCNN](https://github.com/timesler/facenet-pytorch)
- [Yolofacev5](https://github.com/elyha7/yoloface])
