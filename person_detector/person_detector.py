#!/usr/bin/env python3
"""RKNN+YOLOv3 person detector."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from socketserver import ThreadingMixIn
import os
import sys
import threading
import time
import urllib.request

from PIL import Image
import cv2
import numpy as np

from rknn.api import RKNN


LATEST_IMG = None
RKNN_INSTANCE = None
CAPTURE = None
ACTIVITY_FLAG = False


def image_process():
    """Capture and process single frame."""
    global LATEST_IMG

    while ACTIVITY_FLAG:
        rc, img = CAPTURE.read()
        if not rc:
            print("Failed to capture image.")
            continue
        try:
            img = cv2.resize(img, (416, 416))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            outputs = RKNN_INSTANCE.inference(inputs=[img])

            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]
            input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
            input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
            input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)
            input_data = []
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
            boxes, classes, scores = yolov3_post_process(input_data)
            # print('boxes', boxes,)
            # print('classes ', classes)
            print("classes ", classes)
            person_found = False
            if classes is not None:
                for cl in classes:
                    if all_classes[cl] == "person":
                        print("Person detected: 1")
                       
                        person_found = True
                        break
                if not person_found:
                    print("Person detected: 0")
                    
            else:
                print("Person detected: 0")
                print("classes is None")
                
            if boxes is not None:
                draw(img, boxes, scores, classes)

            LATEST_IMG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except KeyboardInterrupt:
            print("Got SIGINT")
            break
        # except KeyboardInterrupt:
        #     server.socket.close()
        #     print("server stop")
        #     capture.release()
        #     rknn.release()
        #     print("video stop")
        #     break
        #     print("all stop")
    if CAPTURE:
        CAPTURE.release()

    RKNN_INSTANCE.release()
    print("video stop")
    print("all stop_2")


class CamHandler(BaseHTTPRequestHandler):
    """HTTP motion jpeg server."""
    def do_GET(self):
        """Handle HTTP GET."""
        if self.path.endswith(".mjpg"):
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=--jpgboundary"
            )
            self.end_headers()
            while True:
                if LATEST_IMG is not None:
                    jpg = Image.fromarray(LATEST_IMG)
                    tmp_file = BytesIO()
                    jpg.save(tmp_file, 'JPEG')
                    self.wfile.write("--jpgboundary\r\n".encode())
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(tmp_file.getbuffer().nbytes))
                    self.end_headers()

                    jpg.save(self.wfile, "JPEG")
                    self.wfile.write('\r\n'.encode())

                    time.sleep(0.05)
                else:
                    self.send_error(404, "err")
                    break
            return

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write("<html><head></head><body>".encode())
        self.wfile.write('<img src="/cam.mjpg"/>'.encode())
        self.wfile.write("</body></html>".encode())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def start_server():
    """Start the HTTP server."""
    global ACTIVITY_FLAG

    # while True:
    #     try:
    server = ThreadedHTTPServer(("0.0.0.0", 8087), CamHandler)
    print("Server started")
    try:
        server.serve_forever()

    except KeyboardInterrupt:
        ACTIVITY_FLAG = False
        server.socket.close()
        server.shutdown()
        print("server stop")
        CAPTURE.release()
        print("capture release")
        RKNN_INSTANCE.release()
        print("video stop")
        print("all stop")


GRID0 = 13
GRID1 = 26
GRID2 = 52
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6

CAP_WIDTH = 768
CAP_HEIGHT = 432
all_classes = (
    "person",
    "bicycle",
    "car",
    "motorbike ",
    "aeroplane ",
    "bus ",
    "train",
    "truck ",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign ",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog ",
    "horse ",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra ",
    "giraffe",
    "backpack",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife ",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza ",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet ",
    "tvmonitor",
    "laptop	",
    "mouse	",
    "remote ",
    "keyboard ",
    "cell phone",
    "microwave ",
    "oven ",
    "toaster",
    "sink",
    "refrigerator ",
    "book",
    "clock",
    "vase",
    "scissors ",
    "teddy bear ",
    "hair drier",
    "toothbrush ",
)


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def process(inputs, mask, anchors):
    """Post-processing helper."""
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, inputs.shape[0:2])

    box_confidence = sigmoid(inputs[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(inputs[..., 5:])

    box_xy = sigmoid(inputs[..., :2])
    box_wh = np.exp(inputs[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= box_wh / 2.0
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    """YOLO v3 inference post-processing."""
    # yolov3
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [
        [10, 13],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
        [116, 90],
        [156, 198],
        [373, 326],
    ]
    # yolov3-tiny
    # masks = [[3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

    boxes, classes, scores = [], [], []
    for inputs, mask in zip(input_data, masks):
        b, c, s = process(inputs, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        if not all_classes:
            print("The CLASSES list is empty.")
        else:

            print("CLASSES contains elements.")

        print(f"class: {all_classes[cl]}, score: {score}")
        print(f"box coordinate left,top,right,down: [{x}, {y}, {x + w}, {y + h}]")
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{all_classes[cl]} {score:.2f}",
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def download_yolov3_weight(dst_path):
    """Download YOLO v3 weights."""
    if os.path.exists(dst_path):
        print("yolov3.weight exist.")
        return
    print("Downloading yolov3.weights...")
    url = "https://pjreddie.com/media/files/yolov3.weights"
    try:
        urllib.request.urlretrieve(url, dst_path)
    except urllib.error.HTTPError as e:
        print("HTTPError code: ", e.code)
        print("HTTPError reason: ", e.reason)
        sys.exit(-1)
    except urllib.error.URLError as e:
        print("URLError reason: ", e.reason)
    else:
        print("Download yolov3.weight success.")


if __name__ == "__main__":

    MODEL_PATH = "./yolov3.cfg"
    WEIGHT_PATH = "./yolov3.weights"
    RKNN_MODEL_PATH = "./yolov3_416.rknn"
    # im_file = "./dog_bike_car_416x416.jpg"
    DATASET = "./dataset.txt"

    # Download yolov3.weight
    # Create RKNN object

    RKNN_INSTANCE = RKNN()

    NEED_BUILD_MODEL = False

    # Direct load rknn model
    print("Loading RKNN model")
    ret = RKNN_INSTANCE.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Load RKNN model failed.")
        sys.exit(ret)
    print("done")

    # Init runtime environment
    print("--> Init runtime environment")
    ret = RKNN_INSTANCE.init_runtime()
    if ret != 0:
        print("Init runtime environment failed.")
        sys.exit(ret)
    print("done")

    # img = cv2.imread(im_file)
    # print(img.shape)

    CAPTURE_PIPE = (
        "libcamerasrc camera-name=/base/i2c@ff110000/ov4689@36 !"
        f"video/x-raw,width={CAP_WIDTH},height={CAP_HEIGHT},format=YUY2 ! "
        "queue leaky=1 max-size-bytes=0 max-size-time=0 max-size-buffers=1 ! "
        "videoconvert ! appsink"
    )

    # global capture
    CAPTURE = cv2.VideoCapture(CAPTURE_PIPE, cv2.CAP_GSTREAMER)
    ACTIVITY_FLAG = True
    threading.Thread(target=image_process, daemon=True).start()
    start_server()
    if RKNN_INSTANCE:
        RKNN_INSTANCE.release()
    # print("Waiting for the first request to start the server...")

    # rknn.release()
