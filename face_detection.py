import cv2
import time
import os

def main():
    # Initialize video capture from the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load the pre-trained face detection classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")
    
    face_classifier = cv2.CascadeClassifier(cascade_path)
    
    # Set desired FPS
    FPS = 30
    frame_time = 1/FPS
    
    try:
        while True:
            start_time = time.time()
            
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Create a copy for drawing
            output = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),  # Blue color in BGR
                    2  # Line thickness
                )
            
            # Display the result
            cv2.imshow('Face Detection', output)
            
            # Calculate how long to wait to maintain desired FPS
            processing_time = time.time() - start_time
            wait_time = max(1, int((frame_time - processing_time) * 1000))
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()