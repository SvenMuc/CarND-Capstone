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
import threading

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image as Img

PATH_TO_SIM_FROZEN_MODEL = "../frozen_inference_graph.pb"
PATH_TO_REAL_FROZEN_MODEL = "../frozen_inference_graph.pb"

PATH_TO_SIM_LABELS = "../traffic_light_label_map.pbtxt"
PATH_TO_REAL_LABELS = "../traffic_light_label_map.pbtxt"

NUM_CLASSES = 4
STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    """ Traffic Light Detector Node. """
    
    def __init__(self):
        """ Initialization. """
        
        rospy.init_node('tl_detector')

        # get parameters from launch file
        self.tl_show_detector_results = rospy.get_param('debug_tl_detector', False)
        
        if self.tl_show_detector_results is True:
            rospy.logwarn('TL detector debugging activated (debug_tl_detector = {}). Please deactivate \
                           it in the launch files for real-time performance.'.format(self.tl_show_detector_results))
            self.bridge = CvBridge()
            self.pub_image_tl_overlay = rospy.Publisher('/image_tl_overlay', Image, queue_size=1)
        else:
            rospy.loginfo('TL detector debugging disabled. You can activate it in the launch files.')

        site_launch = rospy.get_param('site', False)
        if site_launch:
            sub6 = rospy.Subscriber('/image_raw', Image, self.image_cb, queue_size = 1, buff_size=2**24)
        else:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size = 1, buff_size=2**24)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.camera_image = None
        self.bridge = CvBridge()

        self.state = -1
        self.last_state = -2
        self.state_count = 0
        self.state_average = [0,0,0,0]
        
        # Load Model and initialize tensorflow
        rospy.loginfo('Loading the TL model...') 
        model_path = PATH_TO_REAL_FROZEN_MODEL
        self.detection_graph = tfl.Graph()
    
        with self.detection_graph.as_default():
            od_graph_def = tfl.GraphDef()
            with tfl.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tfl.import_graph_def(od_graph_def, name='')
            self.sess = tfl.Session(graph=self.detection_graph)

        rospy.loginfo('TL model loaded.')

        # run tl detector thread
        rospy.loginfo('Start TL detector thread...')
        self.event_has_image = threading.Event()
        self.lock_image = threading.Lock()
        tl_thread = threading.Thread(target=self.process_image)
        tl_thread.start()
        rospy.loginfo('TL detector up and running.')
        
        rospy.spin()

    def load_image_into_numpy_array(self, image):
        """ Image converter ROS Image to numpy array. """
        cv_image = self.bridge.imgmsg_to_cv2(image, "rgb8")
        im_width = image.width
        im_height = image.height
        return np.array(cv_image).reshape(im_height, im_width, 3).astype(np.uint8)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.lock_image.acquire()
        self.camera_image = msg
        self.lock_image.release()
        self.event_has_image.set()

    def process_image(self):
        """" Perform Model Prediction. """

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        while not rospy.is_shutdown():
            # wait for valid image to process
            if not self.event_has_image.wait(1):
                continue
            
            self.lock_image.acquire()
            image = self.camera_image
            self.lock_image.release()
            image_np = self.load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Perform Detection
            time1 = time.time()
            (boxes, scores, classes, num) = self.sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            delta_time = time.time() - time1
            #rospy.logerr('Number of detections: {}'.format(num))
            #rospy.logerr('Classes:')
            #rospy.logerr(classes)
            #rospy.logerr('scores:')
            #rospy.logerr(scores)

            # if delta_time > 1.0:
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

            # rospy.logwarn("Traffic State Prediction: {}".format(tl_state_dict[tl_state_prediction]))
            #rospy.logwarn("Traffic State Confidence: {}".format(scores[0][np.argmax(scores)]))

            # If the recent state was detected 3/4 of the last detections, publish it

            pubmsg = Int32()
            pubmsg.data = tl_state_prediction
            self.upcoming_red_light_pub.publish(pubmsg)

            # state = tl_state_prediction
            # self.state_average.pop(0)
            # self.state_average.append(state)
            # if (self.state_average.count(state) >= 3):
                #rospy.logwarn("Traffic Light State Published: {}".format(tl_state_dict[state])
            
            self.event_has_image.clear()
        
        # clean-up tensorflow session
        self.sess.close()    
        rospy.loginfo('TL detector thread stopped.')

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
