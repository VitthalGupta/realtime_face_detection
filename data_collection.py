import cv2
import os
from path import path, path_dataset, path_users
# Create Local Binary Patterns Histograms for face recognization
cam = cv2.VideoCapture(0)

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

# create directory to the user
#check if the directory exits already
os.chdir(path_users)
x = os.listdir()
if face_id not in x:
    os.mkdir(face_id)
os.chdir(face_id)
print(os.getcwd())

print("Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while (True):

    ret, img = cam.read()
    # img = cv2.flip(img, -1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # time.sleep(2)
    faces = face_detector.detectMultiScale( gray , 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(str(face_id)+ '_' +
                    str(count) + ".png", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("Cleaning up!")
os.chdir(path)
cam.release()
cv2.destroyAllWindows()
