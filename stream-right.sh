#!/bin/bash

gst-launch-1.0 libcamerasrc camera-name=/base/i2c@ff120000/ov4689@36 ! video/x-raw,width=640,height=480,format=NV12 ! videoconvert ! v4l2jpegenc ! rtpjpegpay ! udpsink host=192.168.42.15 port=1234
