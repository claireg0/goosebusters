import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

def main():
    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)
    
    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load the YOLOv8 model
    model = YOLO('yolo11n.pt')
    
    # Define the class ID for birds (including geese) in COCO dataset
    BIRD_CLASS_ID = 14
    
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
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.5)
            
            # Process the results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # print(int(box.cls))
                    # Check if the detected object is a bird
                    if int(box.cls) == BIRD_CLASS_ID:
                        # Get the bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Draw the bounding box with a thicker line
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw center point
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                        
                        # Calculate relative position (0-100%)
                        rel_x = round((center_x / frame_width) * 100)
                        rel_y = round((center_y / frame_height) * 100)
                        
                        # Add confidence score and coordinates
                        conf = float(box.conf)
                        coord_text = f'Goose: {conf:.2f} | Pos: ({x1},{y1})-({x2},{y2})'
                        rel_pos_text = f'Relative: {rel_x}%, {rel_y}%'
                        
                        # Print coordinates to console
                        print(f"\rDetected goose at: {coord_text} | {rel_pos_text}", end='')
                        
                        # Draw text on frame with background
                        # Function to add text with background
                        def put_text_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, 
                                                  scale=0.6, color=(0, 255, 0), thickness=2):
                            # Get text size
                            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
                            
                            # Create background rectangle
                            padding = 5
                            cv2.rectangle(img, 
                                        (pos[0] - padding, pos[1] - text_height - padding),
                                        (pos[0] + text_width + padding, pos[1] + padding),
                                        (0, 0, 0), -1)
                            
                            # Add text
                            cv2.putText(img, text, pos, font, scale, color, thickness)
                        
                        # Add text with background
                        put_text_with_background(frame, coord_text, (x1, y1 - 10))
                        put_text_with_background(frame, rel_pos_text, (x1, y1 - 40))
                        
                        # Draw crosshairs at center point
                        line_length = 20
                        cv2.line(frame, (center_x - line_length, center_y),
                                (center_x + line_length, center_y), (255, 0, 0), 2)
                        cv2.line(frame, (center_x, center_y - line_length),
                                (center_x, center_y + line_length), (255, 0, 0), 2)
            
            # Display the result
            cv2.imshow('Canada Goose Detection', frame)
            
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
        print("\nDetection stopped")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main()