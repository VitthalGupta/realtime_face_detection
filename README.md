"""

# Face Recognition System

This is a Python code for a face recognition system. It allows you to collect face data, train the system with the collected data, and recognize faces in real-time.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- OpenCV library

You can install the required packages using the following command:

```bash
pip install opencv-python
```

## Getting Started

1. Clone the repository or download the code files to your local machine.

2. Update the configuration (config.ini) file:
   - The configuration file (`config.ini`) is automatically created if it doesn't exist.
   - It contains following options:
     - `face_id_count`: The current count of face IDs. Do not modify this manually.
     - `face_samples`: The number of face samples to collect for training.Modify this value according to your needs.
     - The `face_id_count` option is used to assign a unique ID to each user. The ID is incremented by 1 for each new user.
     - A relation is established between the user ID and the user name in the `face_labels` dictionary.


3. Collect Face Data:
   - The script `data_collection.py` to capture face data for a new user.
   - Use the `-n` or `--name` argument followed by the name of the user to create a new user directory.
   - Example: `python data_collection.py -n John`

4. Train the Face Recognition System:
   - After collecting face data for one or more users, the script `face_training.py` is used to train the face recognition model.
   - This step will generate a trained model file (`trainer.yml`) in the `trainer` directory.

5. Recognize Faces:
   - The script `detector.py` is used to start the real-time face recognition system.
   - The system will use the trained model to recognize faces captured by the camera.

## Usage

To create a new user and collect face data:

```bash
python main.py -n [user_name]
```

To train the face recognition system:

```bash
python main.py -t
```

To recognize faces in real-time:

```bash
python main.py -r
```

Note: If no argument is provided, the script will automatically start the face recognition system.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute the code for personal or commercial use.
"""
