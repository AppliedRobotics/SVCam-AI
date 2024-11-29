
import numpy as np
import cv2
import time

CAP_WIDTH = 640
CAP_HEIGHT = 480

if __name__ == "__main__":
    

    #right
    CAPTURE_PIPE_l = (
    "libcamerasrc camera-name=/base/i2c@ff110000/ov4689@36 !"
    f"video/x-raw,width={CAP_WIDTH},height={CAP_HEIGHT},format=YUY2 ! videoconvert ! appsink"
)
    CAPTURE_PIPE_r = (
    "libcamerasrc camera-name=/base/i2c@ff120000/ov4689@36 !"
    f"video/x-raw,width={CAP_WIDTH},height={CAP_HEIGHT},format=YUY2 ! videoconvert ! appsink"
)
    
    global capture_left
    global img_left
    global img_right
    global capture_right

    capture_left = cv2.VideoCapture(CAPTURE_PIPE_l, cv2.CAP_GSTREAMER)
    capture_right = cv2.VideoCapture(CAPTURE_PIPE_r, cv2.CAP_GSTREAMER)
    count = 0
    count_image = 0

    try:
        while True:
            rc_l, img_left = capture_left.read()
            rc_r, img_right = capture_right.read()
            if count % 10 == 0:
                cv2.imwrite("stereo_left/{number}.png".format(number=count_image),img_left)
                cv2.imwrite("stereo_right/{number}.png".format(number=count_image),img_right)
                count_image += 1
            count += 1
            
            print('space for next step')
            #input()
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("keyboard interrupt")


    