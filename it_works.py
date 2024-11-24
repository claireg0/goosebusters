import argparse
import sys
import time
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import AngularServo
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
from picamera2.encoders import JpegEncoder
from collections import deque

from utils import visualize
from servo_pump import set_angle, inc_angle, start_shooting, stop_shooting

COUNTER, FPS = 0, 0
START_TIME = time.time()
#DETECTION_QUEUE = queue.Queue()
DETECTION_QUEUE = deque(maxlen=1)

RELAY_1_GPIO = 26
SERVO_1_GPIO = 11

# Initialize Picamera2 globally to be reused across threads
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"}))

set_angle(135)

def add_to_queue(item):
    DETECTION_QUEUE.append(item)  # Automatically removes the oldest if full

def detection_thread(model: str, max_results: int, score_threshold: float)->None:
    """Object detection thread."""
    global COUNTER, FPS, START_TIME

    # Start the camera
    picam2.start()

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        """Callback to process detection results."""
        global FPS, COUNTER, START_TIME
        if COUNTER % 10 == 0:
            FPS = 10 / (time.time() - START_TIME)
            START_TIME = time.time()
        if result.detections:
            add_to_queue(result)
        COUNTER += 1

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        category_allowlist=["bird"],
        result_callback=save_result
    )
    detector = vision.ObjectDetector.create_from_options(options)

    try:
        while True:
            frame = picam2.capture_array()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            # Display FPS on the frame
            cv2.putText(
                frame,
                f'FPS: {FPS:.1f}',
                (24, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )
            cv2.imshow('object_detection', frame)

            if cv2.waitKey(1) == 27:  # Exit on pressing 'Esc'
                break
    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        detector.close()
        picam2.stop()
        cv2.destroyAllWindows()


def servo_pump_thread():
    try:
        '''
        GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers 
        GPIO.setup(RELAY_1_GPIO, GPIO.OUT) # GPIO Assign mode
        GPIO.output(RELAY_1_GPIO, GPIO.LOW)

        servo = AngularServo(SERVO_1_GPIO, min_angle =0, max_angle = 270, min_pulse_width =0.5/1000, max_pulse_width=2.5/1000)
        servo.detach()
       
        set_angle(135)
        '''
        while True:
            try:
                if not DETECTION_QUEUE:
                    stop_shooting()
                    continue

                result = DETECTION_QUEUE.popleft()  # Wait for detection result
                start_shooting()
                for detection in result.detections:
                    bbox = detection.bounding_box
                    x_center = bbox.origin_x + bbox.width / 2
                    if x_center < 1920 / 2 - 20:
                        inc_angle(0.4)
                    elif x_center > 1920 / 2 + 20:
                        inc_angle(-0.4)
            except Exception as e:
                print(f"Error in servo_pump_thread: {e}")
    except KeyboardInterrupt:
        print("stopped")
        GPIO.cleanup()
    except GPIOPinInUse as e:
        print(f"GPIO Pin conflict: {e}")
        GPIO.cleanup()
        raise
    except Exception as e:
        print(f"Error in servo_pump_thread: {e}")
    finally:
        GPIO.cleanup()
        print("Servo pump thread terminated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet_lite0.tflite', help='Path to detection model.')
    parser.add_argument('--maxResults', default=1, type=int, help='Max detection results.')
    parser.add_argument('--scoreThreshold', default=0.2, type=float, help='Score threshold.')
    args = parser.parse_args()

    # Start threads
    try:
        detection_t = threading.Thread(
            target=detection_thread,
            args=(args.model, args.maxResults, args.scoreThreshold),
        )
        servo_pump_t = threading.Thread(target=servo_pump_thread)

        detection_t.start()
        servo_pump_t.start()

        detection_t.join()  # Wait for threads to finish
        servo_pump_t.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping threads.")
    finally:
        GPIO.cleanup()
        print("Application terminated.")


if __name__ == '__main__':
    main()
