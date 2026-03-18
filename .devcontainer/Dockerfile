FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-yaml \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update &&  apt-get install -y --no-install-recommends \
    ros-humble-ackermann-msgs \
    ros-humble-vision-msgs \
    ros-humble-visualization-msgs \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    ros-humble-image-transport \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*
    
RUN python3 -m pip install --no-cache-dir --upgrade pip

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc