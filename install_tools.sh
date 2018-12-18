#! /bin/bash
set -e

# TensorFlow Serving dependencies
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python3-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
        
# TensorFlow Model Server
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server-universal

# Python packages
pip3 install tensorflow tensor2tensor nltk sacremoses pandas matplotlib networkx grpcio

# NLTK packages
python3 -c "import nltk; nltk.download(['perluniprops', 'punkt'])"

# Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone the slug2slug repo
git clone https://github.com/jjuraska/slug2slug.git
