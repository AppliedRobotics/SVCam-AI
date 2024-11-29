import cv2
import numpy as np


def receive():
    print(cv2.getBuildInformation())
    pipeline2 = 'udpsrc port=1234 ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec !   appsink'
    cap = cv2.VideoCapture(pipeline2, cv2.CAP_GSTREAMER)

    while True:
        ret,frame = cap.read()
        if not ret:
            print('empty frame')
            
            continue
        print(frame.shape)
        cv2.imshow('receive', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()

print(cv2.getBuildInformation())
#receive()