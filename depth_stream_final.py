
import cv2
import numpy as np

def nothing():
    pass

def compute_distance_to_object( depth_map, roi):

        x, y, w, h = roi
        roi_depth_map = depth_map[y:y+h, x:x+w]
        
        # Исключаем нулевые значения из расчета, так как они могут быть результатом отсутствия данных.
        valid_depth_values = roi_depth_map[roi_depth_map > 0]
        
        if valid_depth_values.size > 0:
            mean_distance = np.mean(valid_depth_values)
            return mean_distance
        else:
            return float('inf')

def draw_rectangle(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def put_text(frame, text, position = (50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame

def gsteam_receive():
    pass
    
    
def distance_worker(queue):

    print(cv2.getBuildInformation())
    flag_manual = False
    CAP_WIDTH = 640
    CAP_HEIGHT = 480
    
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
    cv_file = cv2.FileStorage("rectify_map_imx219_160deg_1080p_new.yaml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("map_l_1").mat()
    Left_Stereo_Map_y = cv_file.getNode("map_l_2").mat()
    Right_Stereo_Map_x = cv_file.getNode("map_r_1").mat()
    Right_Stereo_Map_y = cv_file.getNode("map_r_2").mat()
    cv_file.release()
 
    count = 0
    stereo = cv2.StereoSGBM_create()
    gstreamer_pipeline =  "appsrc ! queue ! videoconvert ! video/x-raw,width=640,height=480,format=NV12 ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=192.168.42.15 port=1240 sync=false"
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(gstreamer_pipeline, cv2.CAP_GSTREAMER, 0, 1, (640, 480), True)
    while True:
        count +=1
        ret_left, left_img = capture_left.read()
        ret_right, right_img = capture_right.read()
        #print(right_img.shape)
        imgR_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        imgL_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        Left_nice= cv2.remap(imgL_gray,
              Left_Stereo_Map_x,
              Left_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
        Right_nice= cv2.remap(imgR_gray,
              Right_Stereo_Map_x,
              Right_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
        if flag_manual:
        
            numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
            blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
            P1 = cv2.getTrackbarPos('P1', 'disp')
            P2 = cv2.getTrackbarPos('P2', 'disp')
            disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
            minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')
        else:
            numDisparities = 3 * 16
            blockSize = 8 * 2 + 5
            P1 = 200
            P2 = 400
            disp12MaxDiff = 25
            minDisparity = 3
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setP1(P1)
        stereo.setP2(P2)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
    
        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
        
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        #print(disparity.shape)
        cv2.imwrite('disp.jpg', disparity)
        roi = [320, 220, 80, 80]
        compute_disparity = compute_distance_to_object(disparity,roi)
        f = 850.4 
        b = 0.065
        mean_distance = (b*f)/compute_disparity 
        
        print(mean_distance)
        queue.put(mean_distance)  # Передаем значение в очередь
        
        draw_rectangle(disparity, roi)
        put_text(disparity, str(mean_distance))
        draw_rectangle(left_img, roi)
        put_text(left_img, str(mean_distance))
        
        #disparity = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)
        depth_map_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_vis = depth_map_vis.astype(np.uint8)  
        depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_JET)
        out.write(left_img)

        #cv2.imshow("disp1",disparity)
        #cv2.imshow("left_img",left_img)
        
    
        # Close window using esc key
            
if __name__ == "__main__":
    distance_worker()
