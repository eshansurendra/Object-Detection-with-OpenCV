# Object Detection with OpenCV

This repository contains a simple project aimed at detecting various objects in real-time camera video data using the YOLO (You Only Look Once) pre-trained model along with OpenCV.

## Requirements
- Python 3.10
- OpenCV
- Flask
- YOLOv3 pre-trained weights and configuration file
- COCO names file

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Object-Detection-with-OpenCV.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Download the YOLOv3 pre-trained weights (`yolov3.weights`) and configuration file (`yolov3.cfg`) from the official YOLO website or repository.
2. Download the COCO names file (`coco.names`) which contains the list of object classes that the model was trained on.
3. Place the downloaded files in the project directory.
4. Run the Flask app:
    ```bash
    python app.py
    ```
    This will start the Flask web server.
5. Navigate to `http://localhost:5000` in your web browser to access the object detection interface.
6. Optionally, you can run the object detection script without Flask:
    ```bash
    python mainWithLabels.py
    ```
    This will run the object detection script without the web interface, directly detecting objects from a video feed.

## References
- YOLO: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- OpenCV: [https://opencv.org/](https://opencv.org/)
- Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
