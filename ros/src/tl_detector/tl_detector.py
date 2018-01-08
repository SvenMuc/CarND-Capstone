#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
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
#from google.protobuf import text_format
import logging

#PATH_TO_FROZEN_MODEL = "/home/student/frozen_inference_graph.pb"
PATH_TO_SIM_FROZEN_MODEL = "../frozen_inference_graph.pb"
PATH_TO_REAL_FROZEN_MODEL = "../frozen_inference_graph.pb"


PATH_TO_SIM_LABELS = "../traffic_light_label_map.pbtxt"
PATH_TO_REAL_LABELS = "../traffic_light_label_map.pbtxt"

NUM_CLASSES = 4
STATE_COUNT_THRESHOLD = 3


model_path = PATH_TO_SIM_FROZEN_MODEL
label_path = PATH_TO_SIM_LABELS

detection_graph = tfl.Graph()

# Image Helper Code
def load_image_into_numpy_array(image):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, "rgb8")  #Format that has worked the best so far. bgr8 did not work
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

        # get parameters from launch file
        self.tl_show_detector_results = rospy.get_param('debug_tl_detector', False)
        
        if self.tl_show_detector_results is True:
            rospy.logwarn('TL detector debugging activated (debug_tl_detector = {}). Please deactivate \
                           it in the launch files for real-time performance.'.format(self.tl_show_detector_results))
            self.bridge = CvBridge()
            self.pub_image_tl_overlay = rospy.Publisher('/image_tl_overlay', Image, queue_size=1)
        else:
            rospy.loginfo('TL detector debugging disabled. You can activate it in the launch files.')

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
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size = 1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.processing = False
        self.state = -1
        self.last_state = -2
        self.last_wp = -1
        self.state_count = 0
        self.state_average = [0,0,0,0]
        self.model_loaded = False
        
        # Load Model
        if (self.model_loaded == False):
            rospy.loginfo('Loading the TL model...') 
            model_path = PATH_TO_REAL_FROZEN_MODEL
            #label_path = PATH_TO_REAL_LABELS   #Only used in model training, not needed for implementation
 
            with detection_graph.as_default():
                od_graph_def = tfl.GraphDef()
                with tfl.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tfl.import_graph_def(od_graph_def, name='')

            self.model_loaded = True
            rospy.loginfo('TL model loaded.') 


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

        self.process_image()
        
        # Load image into np array

    def process_image(self):
        ### Perform Model Prediction


        if self.has_image == True:
            image = self.camera_image
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

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
                    delta_time = time.time() - time1
                    #rospy.logerr('Number of detections: {}'.format(num))
                    #rospy.logerr('Classes:')
                    #rospy.logerr(classes)
                    #rospy.logerr('scores:')
                    #rospy.logerr(scores)

                    if delta_time > 1.0:
                        rospy.logwarn('Time for model prediction > 1.0 sec: {} s'.format(delta_time))

                    # Publish TL overlay image for debgging if activated in the launch file
                    if self.tl_show_detector_results is True:
                        rospy.loginfo('Top 10 Classes: {}'.format(classes[0, 0:10]))
                        rospy.loginfo('Top 10 Scores: {}'.format(scores[0, 0:10]))
                        image_np = self.draw_bounding_boxes(image_np, boxes, scores, classes, min_score_thresh=0.5, line_thickness=2)

                        try:
                            self.pub_image_tl_overlay.publish(self.bridge.cv2_to_imgmsg(image_np, "rgb8"))
                        except CvBridgeError as e:
                            print(e)

            # Grab the class with the heighest prediction score
            # 1 = Undefined, 2 = Red, 3 = Yellow, 4 = Green
            score = scores[0][np.argmax(scores)]
            if (score >= .50):
                tl_state_prediction = classes[0][np.argmax(scores)]
            else :
                tl_state_prediction = 1
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
                #rospy.logwarn("Traffic Light State Published: {}".format(tl_state_dict[state]))


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

    def draw_bounding_boxes(self, image, boxes, scores, classes, min_score_thresh=0.5, line_thickness=4):
        """ Draws the bounding boxes as overlay onto the RGB image.

        :param image:            RGB numpy image.
        :param boxes:            List of bounding boxes.
        :param scores:           Detection scores.
        :param min_score_thresh: Minimum score threshold for a box to be visualized. If None draw all boxes.
        :param line_thickness:   Thickness of bounding box lines.
        :param classes:          Traffic light classes (1=undefined, 2=red, 3=yellow, 4=green)

        :return: Image with bounding box overlay.
        """
        height, width, _ = image.shape

        for i in range(len(boxes)):
            for n in range(len(boxes[i])):
                if min_score_thresh is None or scores[i][n] >= min_score_thresh:
                    color = (100, 100, 100)
                    text = 'TL_undefined'

                    if classes[i][n] == 2:
                        color = (255, 0, 0)
                        text = 'TL_red'
                    elif classes[i][n] == 3:
                        color = (255, 255, 0)
                        text = 'TL_yellow'
                    elif classes[i][n] == 4:
                        color = (0, 255, 0)
                        text = 'TL_green'

                    text = '{} {:.1f}%'.format(text, scores[i][n] * 100.0)
                    x_max = int(round(boxes[i][n][3] * width))
                    x_min = int(round(boxes[i][n][1] * width))
                    y_max = int(round(boxes[i][n][2] * height))
                    y_min = int(round(boxes[i][n][0] * height))

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=line_thickness)
                    cv2.putText(image, text,
                                (x_min, y_min - line_thickness - 1),
                                cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.4,
                                color=color,
                                lineType=1)
        return image


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
