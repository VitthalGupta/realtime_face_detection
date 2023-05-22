import sys
sys.path.append('/Users/vitthal/Documents/GitHub/realtime_face_detection/Codes')
import argparse

# from data_collection import create_user_directory, capture_faces
# from face_training import train_faces
from detector import recognize_faces_object

if __name__ == '__main__':

    # face_id = input('\n enter user id end press <return> ==>  ')
    # create_user_directory(face_id)
    # capture_faces(face_id)
    # train_faces()
    recognize_faces_object()