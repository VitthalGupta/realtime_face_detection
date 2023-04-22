# Realtime Face Detection using OpenCV and Python

## Create Local Binary Patterns Histograms for face recognition

This Python script creates Local Binary Patterns Histograms (LBPH) for face recognition. It captures face images from a video stream, detects faces using Haar cascades, and saves the images in a dataset folder. The script then trains a LBPH model with the saved dataset and stores it in the trainer folder.

## Requirements

* Python 2.7
* OpenCV
* Numpy
* Pillow

## Usage

Run the following command:

```bash
    python face_detection.py
```

* Capture face images using the script
* Train the LBPH model using the captured images
* Use the trained model for face recognition

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
