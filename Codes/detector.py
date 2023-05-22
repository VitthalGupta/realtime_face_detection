import cv2
import os
import numpy as np
from path import path, path_faces,path_classifier, path_trainer, path_yolov3, path_ssd

def recognize_faces_object():
    recognizer = cv2.face.createEigenFaceRecognizer()
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

            if nbr_predicted == 1:
                nbr_predicted = 'Vitthal'
            elif nbr_predicted == 2:
                nbr_predicted = 'Shaurya'
            elif nbr_predicted == 3:
                nbr_predicted = 'Pragyan'
            else:
                nbr_predicted = 'Unknown'

            cv2.putText(im, str(nbr_predicted) + "--" + str(conf),
                        (x, y+h), font, 1.1, (0, 255, 0))  # Draw the text
            cv2.imshow('im', im)
            cv2.waitKey(10)


# print(cv2.__version__)
# def recognize_faces_object():
#     recognizer = cv2.face.EigenFaceRecognizer_create()
#     recognizer.read(os.path.join(path_trainer, 'trainer.yml'))
#     # recognizer = face_recognition.load_model(os.path.join(path_trainer, 'trainer.yml'))

#     cascadePath = os.path.join(path_classifier, 'face.xml')
#     faceCascade = cv2.CascadeClassifier(cascadePath)

#     # Load SSD network
#     net = cv2.dnn.readNetFromCaffe(
#         os.path.join(path_ssd, 'deploy.prototxt'),
#         os.path.join(path_ssd, 'res10_300x300_ssd_iter_140000.caffemodel')
#     )
#     classes = ['background', 'face']

#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     while True:
#         ret, im = cam.read()

#         # Perform object detection using SSD
#         blob = cv2.dnn.blobFromImage(
#             cv2.resize(im, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
#         )
#         net.setInput(blob)
#         detections = net.forward()

#         height, width = im.shape[:2]

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence > 0.5:
#                 class_id = int(detections[0, 0, i, 1])

#                 if class_id != 1:  # Skip non-face detections
#                     continue

#                 x1 = int(detections[0, 0, i, 3] * width)
#                 y1 = int(detections[0, 0, i, 4] * height)
#                 x2 = int(detections[0, 0, i, 5] * width)
#                 y2 = int(detections[0, 0, i, 6] * height)

#                 face_img = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
#                 face_img = cv2.resize(face_img, (100, 100))

#                 nbr_predicted, conf = recognizer.predict(face_img)
#                 cv2.rectangle(im, (x1, y1), (x2, y2), (225, 0, 0), 2)

#                 if nbr_predicted == 1:
#                     nbr_predicted = 'Vitthal'
#                 elif nbr_predicted == 2:
#                     nbr_predicted = 'Shaurya'
#                 elif nbr_predicted == 3:
#                     nbr_predicted = 'Pragyan'

#                 cv2.putText(im, str(nbr_predicted) + "--" + str(conf),
#                             (x1, y2), font, 1.1, (0, 255, 0), 2)

#         cv2.imshow('im', im)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cam.release()
#     cv2.destroyAllWindows()

# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1]
#                      for i in net.getUnconnectedOutLayers()]
#     return output_layers



