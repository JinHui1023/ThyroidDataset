# 使用NVIDIA官方基础镜像
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装PyTorch for Jetson
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 安装TensorRT
RUN apt-get update && apt-get install -y \
    tensorrt \
    python3-libnvinfer \
    python3-libnvinfer-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app
COPY . .

# 设置容器入口
CMD ["python3", "main.py"]
