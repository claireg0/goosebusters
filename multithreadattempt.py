import argparse
import sys
import time
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize
from servo_pump import set_angle, inc_angle, start_shooting, stop_shooting

COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_QUEUE = queue.Queue()

RELAY_1_GPIO = 26
SERVO_1_GPIO = 11

def detection_thread(model: str, max_results: int, score_threshold: float, camera_id: int, width: int, height: int)->None:
    """Object detection thread."""
    global COUNTER, FPS, START_TIME

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        """Callback to process detection results."""
        global FPS, COUNTER, START_TIME
        if COUNTER % 10 == 0:
            FPS = 10 / (time.time() - START_TIME)
            START_TIME = time.time()
        if result.detections:
            DETECTION_QUEUE.put(result)
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

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam.')
        rgb_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Display FPS
        cv2.putText(
            image,
            f'FPS: {FPS:.1f}',
            (24, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        cv2.imshow('object_detection', image)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

def servo_pump_thread():
    try:
        GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers 
        GPIO.setup(RELAY_1_GPIO, GPIO.OUT) # GPIO Assign mode
        GPIO.output(RELAY_1_GPIO, GPIO.LOW)

        servo = AngularServo(SERVO_1_GPIO, min_angle =0, max_angle = 270, min_pulse_width =0.5/1000, max_pulse_width=2.5/1000)
        servo.detach()
        set_angle(135)

        while True:
            try:
                result = DETECTION_QUEUE.get(timeout=1)  # Wait for detection result
                start_shooting()
                for detection in result.detections:
                    bbox = detection.bounding_box
                    x_center = bbox.origin_x + bbox.width / 2
                    if x_center < 320 / 2 - 3:
                        inc_angle(3)
                    elif x_center > 320 / 2 + 3:
                        inc_angle(-3)
            except queue.Empty:
                stop_shooting()
    except KeyboardInterrupt:
        print("stopped")
        GPIO.cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet_lite0.tflite', help='Path to detection model.')
    parser.add_argument('--maxResults', default=1, type=int, help='Max detection results.')
    parser.add_argument('--scoreThreshold', default=0.2, type=float, help='Score threshold.')
    parser.add_argument('--cameraId', default=0, type=int, help='Camera ID.')
    parser.add_argument('--frameWidth', default=320, type=int, help='Frame width.')
    parser.add_argument('--frameHeight', default=320, type=int, help='Frame height.')
    args = parser.parse_args()

    # Start threads
    detection_t = threading.Thread(
        target=detection_thread,
        args=(args.model, args.maxResults, args.scoreThreshold, args.cameraId, args.frameWidth, args.frameHeight),
        daemon=True
    )

    servo_pump_t = threading.Thread(target=servo_pump_thread, daemon=True)

    detection_t.start()
    servo_pump_t.start()

    detection_t.join()  # Main thread waits for detection thread to complete
    servo_pump_t.join()

    GPIO.cleanup()


if __name__ == '__main__':
    main()