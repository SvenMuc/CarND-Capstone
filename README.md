## Capstone Project for Carla

[//]: # (Image References)
[image_pretrained_rfcn]: ./imgs/pre-trained_R-FCN_Bosch.png
[image_generator_label_heatmap_all]: ./tl_model/images/generator_label_heatmap_all.png
[image_generator_label_heatmap_rygo]: ./tl_model/images/generator_label_heatmap_red_yellow_green_off.png
[image_result_real]: ./imgs/capstone_real_augmented.gif
[image_result_sim]: ./imgs/capstone_sim_augmented.gif
[image_timings]: ./imgs/r-fcn_timings.png
[team_members_emails]: ./imgs/final_project_emails.PNG
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Team Members
![Team Members and Emails][team_members_emails]


### Project Overview
This project consists of ROSNodes that collect telemetry and camera data in order to drive through waypoints on a road.
The system adapts to changing traffic patterns in order to have the vehicle arrive safely at it's destination.



### Project Architecture
- Nodes List
    - subscribes
    - Publishes
    - Description

### Decision Reasoning

#### Traffic Light Detection and Classification

The traffic light (TL) detection and classification task can basically be realized by traditional computer vision algorithms as described e.g. in this paper [Semantic segmentation based traffic light detection at day and at night from Vladimir Haltakov,, Jakob Mayr, Christian Unger, and Slobodan Ilic](http://campar.in.tum.de/pub/haltakov2015gcpr/haltakov2015gcpr.pdf) or by Deep Neural Network (DNN) approaches. The latter one becomes very successful in public object detection challenges. In June 2017 Google released the [Tensorflow Object-Detectino API](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html) with a couple of [pre-trained DNN models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) which outperforms almost all conventional approaches.

The paper [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf) gives a good overview about suitable network architectures like SSD, Fast R-CNN, Faster R-CNN and R-FCN and compares each of them regarding speed and accuracy. The TL detector shall be able to run in realtime on CARLA. Due to this restriction we decided to implement the Region-based Fully Convolutional Network (R-FCN) with ResNet-101 for the feature extractor which is a good compromise between speed and accuracy. Details about the model can be read in the paper [R-FCN: Object Detection via Region-based Fully Convolutional Networks, Jifeng Dai Yi (Microsoft Research), Li∗ Kaiming (Tsinghua University), He Jian Sun (Microsoft Research)](https://arxiv.org/pdf/1605.06409.pdf).

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

The individual datasets are extremely unbalanced. In the Bosch small traffic light dataset the labeled traffic lights are almost located in the top right quarter of the image, whereas in the Udacity site ROS bag they are located 100% on the left half of the image. Furthermore, the size of the traffic lights and class distribution (red, yellow, green, undefined) vary a lot between the datasets.

To balance the dataset, 65% of the whole dataset has been augmented by the following methods. Each  augmentation method has been combined by its own probability.

- random translation: tx_max=+/-70, ty_max=+70, probability=20%
- random horizontal flip, probability=50%
- random brightness, probability=20%

For the model training process we implemented a generator which theoretically generates a endless number of images with random augmentation. The following graphs show the total and the individual class label (red, yellow, green and undefined) position distribution after image augmentation. In total we generated 15000 images for the model training and split them into a 90% training and 10% validation dataset.

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

**Timings**

The final R-FCN Traffic Light model has been froozen and pruned and thus is able to run in realtime with **mean=61.19 ms** (min=59.54ms, max=65.12ms) on a PC with NVIDIA GPU (AMD Ryzen 7 1700 8-CoreProcessor, NVIDA GeForce GTX1080 Ti). On a actual CPU (3,1 GHz Intel Core i7) the model needs around 2 s to process one image.

The box plot below summarizes the timings we need to process one image with different image resolutions. We achieve best results (mean=61.19 ms, min=59.54ms, max=65.12ms) with a resolution of 1368x1096 which is exactly the resolution of the camera used in CARLA.

![R-FCN Timings][image_timings]

### Dependencies

### Trouble Shooting


### How DBW Works?
We have created a TwistController class from twist_controller.py which will be used for implementing the necessary controllers. The throttle values passed to publish should be in the range 0 to 1, although a throttle of 1 means the vehicle throttle will be fully engaged. Brake values passed to publish should be in units of torque (N*m). The correct values for brake can be computed using the desired acceleration, weight of the vehicle, and wheel radius.

Here we will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities. Yawcontroller is used to convert target linear and angular velocity to steering commands. `/current_velocity` provides the velocity of the vehicle from simulator.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

Once we have the proposed throttle, brake, and steer values, we published it on the various publishers.
We have currently set up to publish steering, throttle, and brake commands at 50hz.  The DBW system on Carla expects messages at this frequency, and will disengage (reverting control back to the driver) if control messages are published at less than 10hz. This is a safety feature on the car intended to return control to the driver if the software system crashes.

### Export Images from Simulator and ROS Bags

The image export node `tl_image_extractor.py` can be configured by its launch files. There are basically two launch files, one for
the simulator setup `tl_image_extractor.launch` and one for the rosbag setup `tl_image_extractor_site.launch`.

**Attention:**
If you have resource limitations on your PC, ensure to deactivate the OpenCV image visualization by setting
`export_show_image` to `False` in both launch files.

**Parameters**
```
<param name="export_directory" type="str" value="/home/student/CarND-Capstone/export"/>
<param name="export_filename" type="str" value="tfl_"/>
<param name="export_rate" type="int" value="1"/>
<param name="export_encoding" type="str" value="bgr8"/>
<param name="export_show_image" type="bool" value="True"/>
```

**Simulator**
1. Check if the export directory (`export_directory`) exists and is empty. The exporter overrides existing images!
2. Start the image extractor node with styx support by `roslaunch launch/styx_image_extractor.launch`
3. Run the simulator
4. Activate camera output in simulator

**ROS Bags**
1. Check if the export directory (`export_directory`) exists and is empty. The exporter overrides existing images!
2. Start the image extractor node by `roslaunch launch/site_image_extractor.launch`
3. Run ROS bag player by `rosbag play ./bags/just_traffic_light.bag`

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

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
