FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 apt-utils -y
RUN apt-get install cmake  build-essential git -y

ENV PATH="/usr/bin/cmake/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
ENV TCNN_CUDA_ARCHITECTURES="61"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils  build-essential git curl ca-certificates \
        wget vim pkg-config unzip rsync cmake \
        ninja-build x11-apps 


# Set working directory
WORKDIR /opt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip --no-cache-dir install --upgrade ipython && \
    pip --no-cache-dir install \
        numpy \
        scipy \
        tqdm \
        lpips \
        mmcv \
        imageio \
        imageio-ffmpeg \
        opencv-python  


ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin

RUN git clone --recurse-submodules -j8 https://github.com/NVlabs/tiny-cuda-nn.git \
        && cd /opt/tiny-cuda-nn/bindings/torch \
        && python setup.py install

WORKDIR /




