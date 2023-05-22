import os

path = os.getcwd()
path_dataset = os.path.join(path, 'dataset')
path_faces = os.path.join(path_dataset, 'faces')
path_models = os.path.join(path, 'model')
path_classifier = os.path.join(path_models, 'Classifier')
path_trainer = os.path.join(path_models, 'trainer')
path_yolov3 = os.path.join(path_models, 'yolov3')
path_ssd = os.path.join(path_models, 'ssd')