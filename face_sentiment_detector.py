# sentiment analysis using face detection
# Iain McIntosh 15/05/2025

import cv2
import numpy as np
from fer import FER

def main():
    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the emotion detector
    emotion_detector = FER(mtcnn=True)
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # For each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get face region for emotion detection
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotions = emotion_detector.detect_emotions(face_roi)
            
            if emotions:
                # Get dominant emotion
                dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                emotion_text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}"
                
                # Display emotion text above the face rectangle
                cv2.putText(frame, emotion_text, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Display the frame
        cv2.imshow('Face Sentiment Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 