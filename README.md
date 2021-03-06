## Capstone Project for Carla

[//]: # (Image References)
[image_pretrained_rfcn]: ./imgs/pre-trained_R-FCN_Bosch.png
[image_generator_label_heatmap_all]: ./tl_model/images/generator_label_heatmap_all.png
[image_generator_label_heatmap_rygo]: ./tl_model/images/generator_label_heatmap_red_yellow_green_off.png
[image_result_real]: ./imgs/capstone_real_augmented.gif
[image_result_sim]: ./imgs/capstone_sim_augmented.gif
[image_timings]: ./imgs/r-fcn_timings.png
[team_members_emails]: ./imgs/final_project_emails.PNG
[system_architecture]:./imgs/System_Architecture.PNG
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

#### Team Members
![Team Members and Emails][team_members_emails]

### Appendix
1. Project Overview
2. Project Architecture
3. Traffic Light Model
4. Dependencies

### Project Overview

This project uses ROS and python in order to implement a waypoint follower for an autonomous vehicle. The vehicle uses an
image recognition model developed with tensorflow to detect traffic lights from a dashboard mounted camera, and will adapt
behaviour accordingly. The vehicle has a maximum set speed of 25 mp/h, and can be switched back and forth from manual
to autonomous driving.

Additionally, the code was developed and tested on a simulator, and will be further tested on Udacity's self driving
vehicle, Carla.

The project was developed on Ubunutu 16.04, and ROS Kinetic Kame


### Project Architecture

The code follows the below architecture (image provided by Udacity)
![System Architecture][system_architecture]

As evidenced in the image, the code is broken into **Perception**, **Planing**, and **Control** blocks which receive data from
and push data to a Car/Simulator.

#### *Perception*
 The perception block is responsible for observing and processing data from the environment around the vehicle. It consists
 largely of two nodes, **Traffic Light Detection** and **Object Detection**, which are responsible for using the dashboard
 camera (dashcam) to detect relevant objects.

##### Traffic Light Detection
 Traffic light detection is responsible for detecting and classifying traffic lights in images provided by the dashcam.
 These images are given a classification of Red/Yellow/Green/None, and an accompanying confidence score. If a Red or Yellow
 light is detected in the image with a sufficient confidence score, a message is published to our Waypoint Updater node.
 More on our method of Traffic Light Detection in the Decisions and Reasoning section.

##### Obstacle detection
 Obstacle detection is currently unimplemented, though could be implemented by broadining the number of classes detected
 by the image processing model.

#### *Planing*
 The planning block is responsible to determining the path and speed of the vehicle. An orriginal course is plotted from
 starting point to destination, and is updated in real time based off information received from the Perception block. This
 is done using the **Waypoint Loader** and **Waypoint Updater** Nodes.

##### Waypoint Loader
 Waypoint loader is responsible for retrieving the intial waypoints from a file stored in the /data directory. The file
  retrieved is influenced by the launch file. Additionally, this functionality could be applied to waypoint
  files generated for custom paths.

  Waypoint loader is only sends the waypoints once, and has no function after that.

##### Waypoint Updater
 Waypoint updater is responsible for determining the next 50 waypoints the vehicle should travel to, and at what speeds
 the vehicle should be at for each of those waypoints. This is done by taking in localization information from the car/simulator
 and traffic light data from the *Perception* block; this data is then processed in order to determine at which waypoints
 the car should stop next (if it should stop), and at what speeds should the car be going at preceding waypoints to stop
 safely and comfortably. These waypoints are then passed to the **Control** block by publishing to the /final_waypoints topic.

#### *Control*
 The control block is responsible for interfacing the vehicle Drive By Wire (DBW) commands and the desired waypoint locations
 and speeds. This block consists of the **DBW* and **Waypoint Follower** Nodes.

##### DBW Node
 DBW is responsible for taking in the current velocity of the car, current car position, suggested linear velocity, suggested
 angular velocity, and will use these to determine what throttle, steering, and brake commands to publish (which are picked up
 by the simulator, or real vehicles actuators/electronic controls system).

 The throttle values passed to publish should be in the range 0 to 1 (a throttle of 1 means the vehicle throttle will be fully engaged).
 Brake values passed to publish should be in units of torque (N*m). The correct values for brake were computed using
 the desired acceleration, weight of the vehicle, and wheel radius.

##### Waypoint Follower
 Waypoint follower node is a C++ program written by the firm Autoware. It is responsible for taking in a list of desired
 waypoints and velocities at those waypoints, and publishing the linear and angular velocity the car should achieve to meet
 those waypoints.

### Traffic Light Model

The traffic light (TL) detection and classification task can basically be realized by traditional computer vision algorithms
as described e.g. in this paper [Semantic segmentation based traffic light detection at day and at night from Vladimir Haltakov,, Jakob Mayr, Christian Unger, and Slobodan Ilic](http://campar.in.tum.de/pub/haltakov2015gcpr/haltakov2015gcpr.pdf)
or by Deep Neural Network (DNN) approaches, with the latter being very successful in recent public object detection challenges.
In fact, in June 2017 Google released the [Tensorflow Object-Detection API](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
with a couple of [pre-trained DNN models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
which outperforms almost all conventional approaches. Due to the recent succes of DNN's for image classification, and
the availability of resources, we chose to persue a DNN method for our Traffic Light Classifier.



The paper [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf)
gives a good overview about suitable network architectures like SSD, Fast R-CNN, Faster R-CNN and R-FCN and compares each
of them regarding speed and accuracy. The TL detector shall be able to run in realtime on CARLA. Due to this restriction
we decided to implement the Region-based Fully Convolutional Network (R-FCN) with ResNet-101 for the feature extractor
which is a good compromise between speed and accuracy. Details about this model can be found in the paper
[R-FCN: Object Detection via Region-based Fully Convolutional Networks, Jifeng Dai Yi (Microsoft Research), Li∗ Kaiming (Tsinghua University), He Jian Sun (Microsoft Research)](https://arxiv.org/pdf/1605.06409.pdf).

We started with a pre-trained R-FCN ResNet-101 model from the [Tensorflow Object-Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The model has been pre-trained on the [COCO](http://cocodataset.org) dataset and is already able to detect traffic lights on the Bosch Small Traffic Light dataset as shown in the image below.

![Pre-trained R-FCN][image_pretrained_rfcn]

As a next step we change the output layer to our desired classes and applied a transfer learning process. The detailed learning process is described in the [readme.md](tl_model/README.md) file.

| ID | Class        |
|:---|:-------------|
| 1  | TL_undefined |
| 2  | TL_red       |
| 3  | TL_yellow    |
| 4  | TL_green     |

#### Used Datasets

We applied 3 different datasets. Further details like the individual label position distribution, type of labels, etc. are described in the [readme.md](tl_model/README.md) file in chapter "Datasets".

| Dataset                                          | Content                                                                                                                                                                                                                  |
|:-------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bosch Small Traffic Light Dataset                | - 5093 images<br>- 10756 labeled traffic lights<br>- Image size: 1280x720x3<br>- Image format: 8 bit RGB, reconstructed from RIIB 12 bit (red-clear-clear-blue)<br>- Source: https://hci.iwr.uni-heidelberg.de/node/6132 |
| Exported images from the Udacity ROS bag         | - 159 images<br>- 159 labeled traffic lights<br>- Image size: 1368x1096x3<br>- Image format: 8 bit RGB                                                                                                                   |
| Exported images from he Udacity Term 3 simulator | - 277 images<br>- 670 labeled traffic lights<br>- Image size: 800x600x3<br>- Image format: 8 bit RGB                                                                                                                     |

#### Data Augmentation

The individual datasets are extremely unbalanced. In the Bosch small traffic light dataset the labeled traffic lights are
concentrated in the top right quarter of the image, whereas in the Udacity site ROS bag they are located 100% on the left half of the image.
Furthermore, the size of the traffic lights and class distribution (red, yellow, green, undefined) were inconsistent between the datasets.

To balance the dataset, 65% of the whole dataset has been augmented by the following methods. Each augmentation was exclusively
performed with the listed probability.

|   Name    |  Probability |
|:----------|:---------------|
| Random Translation: tx_max=+/-70, ty_max=+70 | 20%|
| Random Horizontal flip | 50%|
| Random Brightness | 20%|


#### Training
For the model training process we implemented a generator which theoretically generates a endless number of images with
random augmentation. The following graphs show the total and the individual class label (red, yellow, green and undefined)
position distribution after image augmentation. In total we generated 15000 images for the model training and split them
into a 90% training and 10% validation dataset.

![Generator Label Heatmnap All][image_generator_label_heatmap_all]

![Generator Label Heatmnap Red, Yellow, Green, Off][image_generator_label_heatmap_rygo]

#### Achieved TL R-FCN Model Performance

For the performance measurement we used the split validation dataset and calculated the Average Precision (AP@IOU0.5) for each class (red, yellow and green) and the Mean Average Precision (mAP@0.5IOU) over all classes according the [PASCAL VOC 2017](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf) definition.

The AP and mAP values are calculated based on the Intersection-of-Union (IOU), number of True Positives (TP) and number of False Positives (FP) and thus do not consider the False Negatives (FN). Our actual model achieves high precision values but sporadically do not detect yellow traffic lights for 1-2 frames.

**Udacity Site**

| Class               | Precision  |
|:--------------------|:-----------|
| AP@0.5IOU TL_red    | 1.0000     |
| AP@0.5IOU TL_yellow | 1.0000     |
| AP@0.5IOU TL_green  | 1.0000     |
| **mAP@0.5IOU**      | **1.0000** |

![TL Detection Real][image_result_real]

**Udacity Simulator**

| Class               | Precision  |
|:--------------------|:-----------|
| AP@0.5IOU TL_red    | 0.9831     |
| AP@0.5IOU TL_yellow | 1.0000     |
| AP@0.5IOU TL_green  | 1.0000     |
| **mAP@0.5IOU**      | **0.9944** |

![TL Detection Sim][image_result_sim]

#### Processing Timings

The final R-FCN Traffic Light model has been frozen and pruned and thus is able to run in realtime with **mean=61.19 ms** (min=59.54ms, max=65.12ms) on a PC with NVIDIA GPU (AMD Ryzen 7 1700 8-CoreProcessor, NVIDA GeForce GTX1080 Ti). On a actual CPU (3.1 GHz Intel Core i7) the model needs around 2 s to process one image.

The box plot below summarizes the timings needed to process images with different resolutions. We achieve best results
(mean=61.19 ms, min=59.54ms, max=65.12ms) with a resolution of 1368x1096 which is exactly the resolution of the camera
used in CARLA.

![R-FCN Timings][image_timings]


#### Export Images from Simulator and ROS Bags

The image export node `tl_image_extractor.py` can be configured by its launch files and will export images from the udacity
simulator. There are basically two launch files, one for the simulator setup `tl_image_extractor.launch` and one for the
rosbag setup `tl_image_extractor_site.launch`. Follow the **Parameters** guide below to assist in setting up the launch file.

##### **Attention:** If you have resource limitations on your PC, ensure to deactivate the OpenCV image visualization by setting `export_show_image` to `False` in both launch files.

##### Parameters
```
<param name="export_directory" type="str" value="/home/student/CarND-Capstone/export"/>
<param name="export_filename" type="str" value="tfl_"/>
<param name="export_rate" type="int" value="1"/>
<param name="export_encoding" type="str" value="bgr8"/>
<param name="export_show_image" type="bool" value="True"/>
```

##### Simulator
1. Check if the export directory (`export_directory`) exists and is empty. The exporter overrides existing images!
2. Start the image extractor node with styx support by `roslaunch launch/styx_image_extractor.launch`
3. Run the simulator
4. Activate camera output in simulator

##### ROS Bags
1. Check if the export directory (`export_directory`) exists and is empty. The exporter overrides existing images!
2. Start the image extractor node by `roslaunch launch/site_image_extractor.launch`
3. Run ROS bag player by `rosbag play ./bags/just_traffic_light.bag`


## Dependencies

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

