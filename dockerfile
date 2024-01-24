# This dockerfile is used to build the deepsniff image.
# It contains the dependencies needed to run deepsniff.

# The base image is the official CUDA image from NVIDIA.


FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# make sure updating the packages is non-interactive

RUN echo e
ENV DEBIAN_FRONTEND=noninteractive

# Install python 3.10.10, pip and cv2 dependencies.

RUN apt-get update && apt-get install -y python3.10 python3-pip ffmpeg libsm6 libxext6  -y

#Create a directory for the deepsniff source code.

#WORKDIR /workspace/deepsniff

# Copy the deepsniff source code to the container.

COPY requirements.txt .

# Install the dependencies.

RUN pip3 install -r requirements.txt


