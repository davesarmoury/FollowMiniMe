#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl

import torch.backends.cudnn as cudnn
from threading import Lock, Thread
from time import sleep

import rospy
from geometry_msgs.msg import PoseStamped, Twist

sys.path.insert(0, '../yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync

lock = Lock()
new_data = False
exit_signal = False

network_width = 540
network_height = 540
image_width = 1920
image_height = 1080

follow_distance = 1
max_vel = 1.5
max_yaw = 1.5


def zed_thread(svo_filepath=None):

    global image_left, image_depth, point_cloud, exit_signal, new_data, camera_info

    rospy.loginfo("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.depth_maximum_distance = 20
    init_params.camera_image_flip  = sl.FLIP_MODE.ON ## Camera mounted upside-down

    runtime_params = sl.RuntimeParameters(sensing_mode=sl.SENSING_MODE.FILL, enable_depth=True)

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        rospy.loginfo(repr(status))
        exit()

    camera_info = zed.get_camera_information().calibration_parameters.left_cam

    image_left_tmp = sl.Mat()
    image_depth_tmp = sl.Mat()
    point_cloud = sl.Mat()

    rospy.loginfo("Initialized Camera")

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve image
            lock.acquire()

            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            zed.retrieve_image(image_depth_tmp, sl.VIEW.DEPTH)
            #zed.retrieve_measure(measure_depth_tmp)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

            image_left = image_left_tmp.get_data()
            image_depth = image_depth_tmp.get_data()

            lock.release()
            new_data = True

        sleep(0.01)

    rospy.loginfo("Cleaning Up Camera")
    image_depth_tmp.free(sl.MEM.CPU)
    image_left_tmp.free(sl.MEM.CPU)
    zed.close()

    rospy.loginfo("ZED Thread Dead...")

def main():
    global image_left, image_depth, point_cloud, exit_signal, new_data, camera_info

    rospy.init_node('tracker_of_trackers', anonymous=True)
    pub = rospy.Publisher('tracker_positioning', PoseStamped, queue_size=10)
    pub2 = rospy.Publisher('tracker_cmd_vel', Twist, queue_size=10)

    msg = PoseStamped()
    msg.header.frame_id = "left_camera"
    msg.pose.orientation.w = 1.0

    capture_thread = Thread(target=zed_thread, kwargs={'svo_filepath': opt.svo})
    capture_thread.start()

    rospy.loginfo("intializing Network...")

    weights, imgsz = opt.weights, opt.img_size
    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    
    model = attempt_load(weights, device=device, fuse=False)  # load FP32 modelimgsz
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    rospy.loginfo("Running tracker detection ... Press 'Esc' to quit")

    while True:
        if new_data:
            lock.acquire()
            visual_frame = image_left.copy()
            lock.release()
            new_data = False

            net_image = visual_frame.copy()

            net_image = net_image[:,:,:3]
            net_image = cv2.resize(net_image, (network_width, network_height))
            net_image = net_image.transpose((2, 0, 1))
            net_image = np.ascontiguousarray(net_image)

            img = torch.from_numpy(net_image).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_sync()
            pred = model(img, augment=opt.augment)[0]

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_sync()

            t_msg = Twist()

            for i, det in enumerate(pred):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        if conf > 0.85:
                            vertical_scale = image_height / network_height;
                            horizontal_scale = image_width / network_width;
                            p_scaled = [xyxy[0] * horizontal_scale, xyxy[1] * vertical_scale, xyxy[2] * horizontal_scale, xyxy[3] * vertical_scale]
                            h_origin = [(p_scaled[0] + p_scaled[2]) / 2.0, (p_scaled[1] + p_scaled[3]) / 2.0]

                            err, world_pose = point_cloud.get_value(int(h_origin[0]), int(h_origin[1]))
                            
                            x_centered = h_origin[0] - ( image_width / 2 )
                            
                            yaw = -math.sin( x_centered * math.pi / image_width )
                            lat_distance = math.sqrt(world_pose[0] * world_pose[0] + world_pose[1] * world_pose[1])

                            rospy.loginfo(lat_distance)
                            x_vel = math.tanh( lat_distance - follow_distance ) * max_vel

                            t_msg.angular.z = yaw
                            t_msg.linear.x = x_vel

                            msg.pose.position.x = world_pose[0]
                            msg.pose.position.y = world_pose[1]
                            msg.pose.position.z = world_pose[2]
                            msg.header.stamp = rospy.Time.now()
                            pub.publish(msg)

                            cv2.rectangle(visual_frame, (int(p_scaled[0]),  int(p_scaled[1])), (int(p_scaled[2]), int(p_scaled[3])), (0,0,255), 3)

            pub2.publish(t_msg)
            t3 = time_sync()

            rospy.loginfo("Done < " + str(t2 - t1) + "s + " + str(t3 - t2) + "s = " + str(t3 - t1) + "s >")

            cv2.imshow("ZED", cv2.resize(visual_frame, (1280, 720)))

            key = cv2.waitKey(5)
            if key == 27:    # Esc key to stop
                exit_signal = True
                break

            sleep(0.01)

    cv2.destroyAllWindows()
    sleep(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../weights/medium.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
