''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow" 
'''

import cv2
import numpy as np
import os
from path import path_faces, path_trainer


def train_faces():
    path = path_faces

    recognizer = cv2.face.EigenFaceRecognizer_create()
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




# import cv2
# import numpy as np
# from PIL import Image
# import os
# from path import path, path_dataset, path_faces

# # Path for face image database
# path = path_faces

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # function to get the images and label data


# def getImagesAndLabels(path):

#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     faceSamples = []
#     ids = []

#     for imagePath in imagePaths:

#         PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
#         img_numpy = np.array(PIL_img, 'uint8')

#         id = int(os.path.split(imagePath)[-1].split(".")[0])
#         faces = detector.detectMultiScale(img_numpy)

#         for (x, y, w, h) in faces:
#             faceSamples.append(img_numpy[y:y+h, x:x+w])
#             ids.append(id)

#     return faceSamples, ids


# print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# faces, ids = getImagesAndLabels(path)
# recognizer.train(faces, np.array(ids))

# # Save the model into trainer/trainer.yml
# # recognizer.save() worked on Mac, but not on Pi
# recognizer.write('trainer/trainer.yml')

# # Print the numer of faces trained and end program
# print("\n [INFO] {0} faces trained. Exiting Program".format(
#     len(np.unique(ids))))