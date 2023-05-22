import cv2
import os
import numpy as np
from path import path, path_dataset, path_faces


def create_user_directory(face_id):
    os.chdir(path_faces)
    x = os.listdir()
    if face_id not in x:
        os.mkdir(face_id)
    os.chdir(face_id)
    print(os.getcwd())


def capture_faces(face_id):
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

    while count < 50:
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
            print(f"Face image saved: {file_name}")
    else:
        print("No faces captured. Skipping saving images.")

if __name__ == '__main__':
    pass