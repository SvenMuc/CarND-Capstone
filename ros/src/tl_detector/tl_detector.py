#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import cv2
import yaml
import tf
import time
import tensorflow as tfl
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as Img


# Used in Label Mapping
from google.protobuf import text_format
import logging
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

#PATH_TO_FROZEN_MODEL = "/home/student/frozen_inference_graph.pb"
PATH_TO_FROZEN_MODEL = "../frozen_inference_graph.pb"
PATH_TO_LABELS = "/home/student/CarND-Capstone/ros/src/tl_detector/object_detection/tl_model_config/traffic_light_label_map.pbtxt"
NUM_CLASSES = 4
STATE_COUNT_THRESHOLD = 3

### Load Frozen Graph
detection_graph = tfl.Graph()
with detection_graph.as_default():
    od_graph_def = tfl.GraphDef()
    with tfl.gfile.GFile(PATH_TO_FROZEN_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tfl.import_graph_def(od_graph_def, name='')

## Load Label Map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Image Helper Code
def load_image_into_numpy_array(image):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, "rgb8")  #Might want a different Format
    im_width = image.width
    im_height = image.height
    return np.array(cv_image).reshape(im_height, im_width, 3).astype(np.uint8)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = -1
        self.last_state = -2
        self.last_wp = -1
        self.state_count = 0
        self.state_average = [0,0,0,0]

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        # Load image into np array
        image_np = load_image_into_numpy_array(msg)
        image_np_expanded = np.expand_dims(image_np, axis=0)


        ### Perform Model Prediction
        with detection_graph.as_default():
            with tfl.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Perform Detection
                time1 = time.time()
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                #rospy.logerr('Number of detections: {}'.format(num))
                #rospy.logerr('Classes:')
                #rospy.logerr(classes)
                #rospy.logerr('scores:')
                #rospy.logerr(scores)
                rospy.logwarn('Time for model prediction: {}'.format(time.time() - time1))


        # Grab the class with the heighest prediction score
        # 1 = Undefined, 2 = Red, 3 = Yellow, 4 = Green
        tl_state_prediction = classes[0][np.argmax(scores)]
        tl_state_dict = {1:'Undefined', 2:'Red', 3:'Yellow', 4:'Green'}
        rospy.logwarn("Traffic State Prediction: {}".format(tl_state_dict[tl_state_prediction]))
        rospy.logwarn("Traffic State Confidence: {}".format(scores[0][np.argmax(scores)]))

        # If the recent state was detected 3/4 of the last detections, publish it
        state = tl_state_prediction
        self.state_average.pop(0)
        self.state_average.append(state)
        if (self.state_average.count(state) >= 3):
            pubmsg = Int32()
            pubmsg.data = tl_state_prediction
            self.upcoming_red_light_pub.publish(pubmsg)
            rospy.logwarn("Traffic Light State Published: {}".format(tl_state_dict[state]))


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
