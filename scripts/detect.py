#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

class ImageSub:
    def __init__(self):
        self.depth_image = None
        self.rgb_image = None
        self.K = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.camera_info = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.target_pub = rospy.Publisher("/target_coord", Point, queue_size=10)
        self.color_pub = rospy.Publisher("/target_color", Point, queue_size=10)
        self.rate = rospy.Rate(10)

    def image_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def camera_info_callback(self, data):
        # get camera intrinsic matrix values
        self.K = (data.K[0], data.K[4], data.K[2], data.K[5])


def detect_color(lower, upper, frame, color=(255,0,0)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(1)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(max_contour)

        if 5 < r < 400:
            region = frame[int(center[1] - r/2):int(center[1] + r/2),
            int(center[0] - r/2):int(center[0] + r/2)]
            avg_color = np.mean(region, axis=(0, 1), keepdims=True)

            center = (int(x), int(y))
            cv2.circle(frame, center, int(r), avg_color[0,0], 2)

    return frame, center, avg_color[0,0]

# blue threshold
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# green threshold
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 150])

# red
lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])

# orange
lower_orange = np.array([10, 125, 180])
upper_orange = np.array([25, 255, 255])


def main():
    rospy.init_node('detection', anonymous=True)
    image_sub = ImageSub()


    while not rospy.is_shutdown():

        center = None
        target_loc = None

        if image_sub.rgb_image is not None and image_sub.depth_image is not None:

            frame = image_sub.rgb_image
            frame_copy = frame.copy()
            frame, center, avg_color = detect_color(lower_orange, upper_orange, frame_copy)

            if center is not None and image_sub.K is not None:
                depth = image_sub.depth_image
                depth_norm = cv2.normalize(image_sub.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                color_norm = cv2.normalize(image_sub.rgb_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # camera intrinsics
                fx, fy, cx, cy = image_sub.K  # focal length, principal point

                x_px, y_px = center

                z_depth = depth[y_px, x_px]

                z = z_depth * 0.001 # distance in meters
                x = (x_px - cx) * z * (1. / fx)
                y = (y_px - cy) * z * (1. / fy)

                dist = math.sqrt(x * x + y * y + z * z)

                if dist > 0.1: # minimum is 0.2m, any depth values below are 0
                    # don't update target location unless valid depth measurement
                    target_loc = (x, z) # don't need y, we only care about 2d xz-plane

                cv2.circle(color_norm, center, int(5), (255,0,0), 2)
                cv2.circle(depth_norm, center, int(5), (155,0,0), 2)

                x_str = "X: " + str(format(x, '.2f'))
                z_str = "Z: " + str(format(z, '.2f'))

                cv2.putText(color_norm, x_str, (x_px+10, y_px), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.7, (0,0,255), 1, cv2.LINE_AA) 
                cv2.putText(color_norm, z_str, (x_px+10, y_px+20), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.7, (0,0,255), 1, cv2.LINE_AA)

                dist = math.sqrt(x**2 + z**2)
        
                dist_str = "dist:" + str(format(dist, '.2f')) + "m"
                cv2.putText(color_norm, dist_str, (x_px+10, y_px+40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.7, (0,255,0), 1, cv2.LINE_AA)

                cv2.imshow("depth", depth_norm)
                cv2.imshow("color", color_norm)

        
        point = Point()
        color = Point()
        if target_loc is not None:
            # publish target location and color
            point.x = target_loc[0]
            point.z = target_loc[1]
            point.y = 0

            color.x = avg_color[0]
            color.y = avg_color[1]
            color.z = avg_color[2]

        else:
            point.x = 0
            point.y = 0
            point.z = 0

            color.x = 0
            color.y = 0
            color.z = 0

        image_sub.target_pub.publish(point)
        image_sub.color_pub.publish(color)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
