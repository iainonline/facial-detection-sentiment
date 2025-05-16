# Face Detection and Sentiment Analysis

This application uses OpenCV and FER (Facial Expression Recognition) to detect faces in real-time through your webcam and analyze their emotional expressions.

## Requirements

- Python 3.7+
- Webcam
- Required packages (listed in requirements.txt)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python face_sentiment_detector.py
```

2. The application will open your webcam and start detecting faces and emotions in real-time.
3. Press 'q' to quit the application.

## Features

- Real-time face detection with bounding boxes
- Emotion detection and display
- Shows confidence score for detected emotions

## Notes

- The application uses your default webcam (index 0)
- Make sure you have good lighting for better detection
- The emotion detector can recognize multiple emotions including: angry, disgust, fear, happy, sad, surprise, neutral 