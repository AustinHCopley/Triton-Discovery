#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.depth_image = None
        self.rgb_image = None

    def image_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('image_subscriber', anonymous=True)
    image_subscriber = ImageSubscriber()

    while not rospy.is_shutdown():

        if image_subscriber.rgb_image is not None:
            frame = image_subscriber.rgb_image
            # cv2.imshow('preview', frame)
        if image_subscriber.depth_image is not None:
            depth = image_subscriber.depth_image
            cv2.imshow('Depth Image', depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
