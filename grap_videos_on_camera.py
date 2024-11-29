import numpy as np
import cv2
import os
import time

CAP_WIDTH = 640
CAP_HEIGHT = 480
FPS = 60
VIDEO_DURATION = 25  # продолжительность видео в секундах

def create_video_writer(filename, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))

if __name__ == "__main__":
    
    # Создание директорий для хранения видео
    os.makedirs("stereo_left_objects_video", exist_ok=True)
    os.makedirs("stereo_right_objects_video", exist_ok=True)

    CAPTURE_PIPE_l = (
    "libcamerasrc camera-name=/base/i2c@ff110000/ov4689@36 !"
    f"video/x-raw,width={CAP_WIDTH},height={CAP_HEIGHT},format=YUY2 ! videoconvert ! appsink"
)
    CAPTURE_PIPE_r = (           
    "libcamerasrc camera-name=/base/i2c@ff120000/ov4689@36 !"
    f"video/x-raw,width={CAP_WIDTH},height={CAP_HEIGHT},format=YUY2 ! videoconvert ! appsink"
)

    capture_left = cv2.VideoCapture(CAPTURE_PIPE_l, cv2.CAP_GSTREAMER)
    capture_right = cv2.VideoCapture(CAPTURE_PIPE_r, cv2.CAP_GSTREAMER)

    if not capture_left.isOpened():
        print("Error: Could not open left camera pipeline.")
        exit(1)
    
    if not capture_right.isOpened():
        print("Error: Could not open right camera pipeline.")
        exit(1)

    count_video = 0

    try:
        while True:
            # Создаем VideoWriter для левой и правой камеры
            video_left_filename = f"stereo_left_objects_video/video_{count_video}.avi"
            video_right_filename = f"stereo_right_objects_video/video_{count_video}.avi"
            
            writer_left = create_video_writer(video_left_filename, CAP_WIDTH, CAP_HEIGHT, FPS)
            writer_right = create_video_writer(video_right_filename, CAP_WIDTH, CAP_HEIGHT, FPS)
            
            start_time = time.time()
            while (time.time() - start_time) < VIDEO_DURATION:
                rc_l, img_left = capture_left.read()
                rc_r, img_right = capture_right.read()

                if not rc_l or img_left is None:
                    print("Error: Failed to capture image from left camera.")
                    break

                if not rc_r or img_right is None:
                    print("Error: Failed to capture image from right camera.")
                    break

                writer_left.write(img_left)
                print('space for next step')
                writer_right.write(img_right)
            
            writer_left.release()
            writer_right.release()
            count_video += 1

    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        capture_left.release()
        capture_right.release()
        print("Resources released.")
