FROM python:3.8

# Install system dependencies for h5py and other pyproject.toml builds
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    build-essential \
    python3-dev \
    python3-setuptools \
    g++ \
    && apt-get clean

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY ./docker/requirements.txt ./
RUN pip install -r requirements.txt

COPY ./bert-code-new /bert-code-new

CMD mkdir /chinese-roberta-wwm-ext-l12-h768-a12 && mkdir /output_predict && mkdir /test-for-inference

WORKDIR /