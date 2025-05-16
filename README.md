# cfar_cpp - Constant False Alarm Rate
**cfar_cpp** is a ROS2 package implementing Constant False Alarm Rate (CFAR) for forward-looking SONAR (FLS). More info about CFAR algorithms can be found in this [Matlab Tutorial](https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html).
This package is tested against ROS2 Humble. It is not guaranteed to work in other distros.

## Package directory tree
```
.
├── Analysis - for data exploration/analysis scripts
├── cfg - YAML config file for CFAR parameters
├── include 
├── launch
├── msg
└── src
```

## Installation
* Install `cv_bridge` and `image-transport` package dependencies:
```
sudo apt update
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport 
```
* Clone and build the package
```
# Make sure to navigate to your ROS 2 workspace's source directory

# Clone the repository (replace with the actual link)
git clone https://github.com/SAAB-NTU/cfar_cpp.git

# Go back to the workspace root
cd ~/ros2_ws

# Build the workspace
colcon build

```

## Usage
```
# Source the workspace
source install/setup.bash

# Launch cfar_node
ros2 launch cfar_cpp cfar_node launch.py
```
* The output can be seen from RViz2

## Miscellaneous
#Data
A copy is in the Desktop (Cannot easily upload via git)
5th and 12th Feb datasets --> Mosaicking/Mapping ==> Check CFAR quality with respect to original method
19th --> Parent Child Tracking  ==> Check CFAR speed here

#Analysis Folder
Oculus reader (Not the best solution, but a working one nonetheless) --> 
1) Convert Oculus V2 log file to V1 log file, then use the python script to extract the individual images
2) Height_map.ipynb --> Currently check what output do the extracted images give
 -- To include FLS position, which will improving mosaicking quality issues due to localization, (Abu Bakr)
 -- To include angled height estimation methods via trigonometry (Abu Bakr)

