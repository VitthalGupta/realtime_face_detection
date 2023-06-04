import cv2
import numpy as np
import os
from path import path_faces, path_trainer


def train_faces():
    path = path_faces

    recognizer = cv2.face.LBPHFaceRecognizer_create() if cv2.__version__.startswith('4') else cv2.face.createLBPHFaceRecognizer()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):
        faceSamples = []
        ids = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.png'):  # Process only PNG image files
                    filePath = os.path.join(root, file)
                    face_img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

                    try:
                        id = int(file.split('_')[0])
                        # Resize the face image to a fixed size (e.g., 100x100)
                        face_img = cv2.resize(face_img, (100, 100))
                        if face_img is not None:
                            faceSamples.append(face_img)
                            ids.append(id)
                        else:
                            print(f"Empty data for ID: {id}. Skipping...")
                    except Exception as e:
                        print(e)
                        print(f"Invalid file name format: {file}. Skipping...")

        return faceSamples, ids



    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)

    if len(faces) > 0 and len(ids) > 0:
        recognizer.train(faces, np.array(ids))
        os.makedirs(path_trainer, exist_ok=True)  # Create 'trainer' directory if it doesn't exist
        recognizer.write(os.path.join(path_trainer, 'trainer.yml'))  # Save the model into trainer/trainer.yml
        print("\n [INFO] {0} faces trained. Exiting Program".format(
            len(np.unique(ids))))
    else:
        print("Insufficient data to train the recognizer. Make sure the dataset contains multiple samples for each person.")
