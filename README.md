# cfar_cpp - Constant False Alarm Rate
**cfar_cpp** is a ROS2 package implementing Constant False Alarm Rate (CFAR) for forward-looking SONAR (FLS). This is a C++ package at its core, but we provide a convenient Python interface using [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) for rapid prototyping and testing. More instructions on how to access those interfaces will be described in the next sections.

This package is extensively tested on **ROS2 Humble**.

## Constant False Alarm Rate (CFAR)
More info about CFAR algorithms can be found in this [Matlab Tutorial](https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html).

## Pipeline
**cfar_cpp** is a rather simple package as the only required input is the FLS image. While the package is mainly for imaging sonar, other 2D signals output can also be used for testing and validation.

## Package Navigation
This package follows the conventional ROS2 package standards. Additionally, the Analysis directory stores all the related calculations and results.
```
.
├── Analysis - for data exploration/analysis scripts
├── cfg
├── include 
├── launch
├── msg
└── src
```

## Installation
These instructions assume everything is done inside a ROS2 workspace (e.g. `~/ros2_ws/src`). Modify the path to your desired ROS2 workspace.
1. **Clone the package**
```bash
cd ~/ros2_ws/src 
git clone https://github.com/SAAB-NTU/cfar_cpp.git
```
2. **Install dependencies**
* Install the related dependencies:
```
sudo apt update
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  libarmadillo-dev \
  libeigen3-dev \
  libboost-dev \

```
3. **Build the package**
```
# Go back to the workspace root
cd ~/ros2_ws

# Build the workspace
colcon build --symlink-install --packages-select cfar_cpp

# Make sure to source the workspace for usage
source install/setup.bash

```

## Usage
This package provides a Python launch file to launch `cfar_node`. It is linked with `cfg/config.yaml` for the CFAR parameters. The instructions for tuning such parameters are provided.

To launch the `cfar_node`:
```
# Source the workspace
source install/setup.bash

# Launch cfar_node
ros2 launch cfar_cpp cfar_node launch.py
```

## Python Interface
As mentioned earlier, `cfar_cpp` provides a Python interface using `pybind11`, named `pycfar`.
Instructions on how to import and use `pycfar` can be found in Jupyter Notebook `Analysis/binding_test.ipynb`.

Before importing `pycfar`, make sure to export the `PYTHONPATH` to the build directory. This can be done either **inside Jupyter Notebook** or **in the terminal**

### Inside Jupyter Notebook
```python
import sys

lib_path = "/path/to/workspace/build/cfar_cpp/lib"
if lib_path not in sys.path:
    sys.path.append(lib_path)

import pycfar
```
### In the terminal
```bash
export PYTHONPATH="/path/to/workspace/build/cfar_cpp/lib:$PYTHONPATH"
```
Additionally, the user can choose to insert this command into `~/.bashrc`

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
