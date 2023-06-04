import cv2
import os
import numpy as np
import configparser
from path import path, path_dataset, path_faces


def create_user_directory(face_name):
    
    # read user count from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'config.ini'))
    face_id_count = config.getint('FaceRecognition', 'face_id_count')
    face_id_count += 1
    config.set('FaceRecognition', 'face_id_count', str(face_id_count))
    #  associate face_id with face_id_count
    config.set('FaceRecognition', str(face_name), str(face_id_count))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    os.chdir(path_faces)
    x = os.listdir()
    if face_id_count not in x:
        os.mkdir(str(face_id_count))
    os.chdir(str(face_id_count))
    print(os.getcwd())

    # initiating capture of face images
    capture_faces(face_name)


def capture_faces(face_name):
    # Get the face id associated with the face name from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'config.ini'))
    face_id = config.getint('FaceRecognition', str(face_name))
    face_samples = config.getint('FaceRecognition', 'face_samples')

    # Initiating Video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Initializing face capture. Look at the camera and wait ...")
    count = 0
    faces_data = []

    # Variables to store the dimensions of the first detected face
    first_face_w = None
    first_face_h = None

    while count < face_samples:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            if first_face_w is None:
                # Store dimensions of the first detected face
                first_face_w = w
                first_face_h = h

            face_img = cv2.resize(
                gray[y:y+h, x:x+w], (first_face_w, first_face_h))
            faces_data.append(face_img)

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(faces_data) > 0:
        # Save multiple PNG images for each captured face
        for i, face in enumerate(faces_data):
            file_name = str(face_id) + '_' + str(i) + '.png'
            cv2.imwrite(file_name, face)
            # print(f"Face image saved: {file_name}")
    else:
        print("No faces captured. Skipping saving images.")

if __name__ == '__main__':
    pass