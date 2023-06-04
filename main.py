import sys
from path import path
import argparse
import os
import configparser

sys.path.append(os.path.join(path, 'Codes'))

from data_collection import create_user_directory
from face_training import train_faces
from detector import recognize_faces_object

if __name__ == '__main__':

   # creating cofigParser object if it soesn't exist
    os.chdir(path)
    if not os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config.add_section('FaceRecognition')
        config.set('FaceRecognition', 'face_id_count', '0')
        config.set('FaceRecognition', 'face_samples', '50')

        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    # write argparse code here for adding new user
    parser = argparse.ArgumentParser()    
    parser.add_argument("-n", "--name", help="Enter the name of the user")
    args = parser.parse_args()
    face_name = args.name
    # if the argument for new user is passed then in itiate the creation of new user
    if face_name:
        # Function to create new user directory
        create_user_directory(face_name)
        
    # Argeoarse code for training the model
    parser.add_argument("-t", "--train", help="Train the model")
    args = parser.parse_args()
    train = args.train
    # if the argument for training is passed then in itiate the training of model
    if train:
        # Function to train the model
        train_faces()
    # Argeoarse code for recognizing faces
    parser.add_argument("-r", "--recognize", help="Recognize faces")
    args = parser.parse_args()
    recognize = args.recognize
    # if the argument for recognizing faces is passed then in itiate the recognition of faces
    if recognize:
        # Function to recognize faces
        recognize_faces_object()
    # If no argument is provided, the script will automatically start the face recognition system.
    if not (face_name or train or recognize):
        # Function to recognize faces
        recognize_faces_object()