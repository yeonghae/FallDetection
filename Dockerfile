ARG PYTORCH="2.2.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install software-properties-common
RUN apt update && apt install -y software-properties-common

# Add deadsnakes PPA for Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update

RUN apt install -y git vim libgl1-mesa-glx libglib2.0-0 ninja-build libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3.9 python3.9-dev python3.9-distutils wget net-tools zip unzip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py

# Install Python Library
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install opencv-python scipy tqdm natsort openpyxl matplotlib chardet moviepy cython xtcocotools ez_setup pymysql ffmpegcv yacs faiss-gpu tensorboard pika
RUN pip install -U scikit-learn

# Install Tensorflow
RUN pip install tensorflow==2.12.0

# Install MMEngine and MMCV
RUN pip install -U openmim
RUN mim install mmcv==2.1.0
RUN mim install mmdet==3.3.0
RUN mim install mmpose==1.3.1
RUN mim install mmengine

# Change Numpy Version
RUN pip uninstall numpy -y
RUN pip install numpy==1.23.5

# Install Fastapi Package
RUN pip install fastapi uvicorn gunicorn starlette python-multipart==0.0.12

## Set the default command to run when the container starts
#WORKDIR /app
#COPY falldown_module.zip .
#RUN unzip falldown_module.zip
#RUN rm falldown_module.zip

# Start the FastAPI server using Gunicorn with Uvicorn workers
CMD ["gunicorn", "--bind", "0:50004", "server:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120"]

