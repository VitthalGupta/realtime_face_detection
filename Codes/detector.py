import cv2
import os
import numpy as np
import configparser
from path import path, path_faces,path_classifier, path_trainer, path_yolov3, path_ssd

def recognize_faces_object():
    recognizer =  cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(path_trainer, 'trainer.yml'))
    cascadePath = os.path.join(path_classifier, 'face.xml')
    faceCascade = cv2.CascadeClassifier(cascadePath)

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX  # Creates a font

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
            100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            nbr_predicted, conf = recognizer.predict(face_img)
            cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)

            # read face_id from config.ini and associate it with face_name, the target variaable is nbr_predicted
            config = configparser.ConfigParser()
            config.read(os.path.join(path, 'config.ini'))
            for face_name in config['FaceRecognition']:
                if config['FaceRecognition'][face_name] == str(nbr_predicted):
                    nbr_predicted = face_name
                    break
            

            cv2.putText(im, str(nbr_predicted) + "--" + str(conf),
                        (x, y+h), font, 1.1, (0, 255, 0))  # Draw the text
            cv2.imshow('im', im)
            cv2.waitKey(10)
