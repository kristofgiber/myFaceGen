#!/bin/bash

FORCE_INSTALL='false'

while getopts f option; do
    case "${option}"
        in
        f ) FORCE_INSTALL='true' ;;
    esac
done

NEEDS_REBOOT='false'

GCLOUD_USERNAME=$(ls /home)
GCLOUD_USERNAME=$(echo "$GCLOUD_USERNAME" | sed '/ubuntu/d')
HOME_DIR="/home/$GCLOUD_USERNAME"
echo "HOME FOLDER OF USER: $HOME_DIR"

###### Stuff that should only be done on first boot

echo "Checking for CUDA and installing."
if ( ! dpkg-query -W cuda-10-0 || $FORCE_INSTALL ); then
    ### INSTALLING CUDA with NVIDIA driver and libCUDNN7
    echo "Installing CUDA..."
    sudo curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
    sudo dpkg -i ./cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda-10-0 -y
    # Add to path:
    export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-2019.1${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    # Download and Install libcudnn7:
    sudo curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.1.34-1+cuda10.0_amd64.deb
    sudo curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.1.34-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7_7.6.1.34-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.6.1.34-1+cuda10.0_amd64.deb
    # Add to path
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"

    # Additional step recommended by NVIDIA for CUDA to work best:
    # Disable udev rule that brings added memories automatically online with default settings preventing NVIDIA from customizing its settings first:
    sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d
    sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules
    # Additional 3rd party libraries recommended by NVIDIA:
    sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
        libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


    echo "CUDA installation finished."

    _PY_SUFFIX="3"
    PYTHON=python${_PY_SUFFIX}
    PIP=pip${_PY_SUFFIX}


    sudo apt-get update
    sudo apt-get install -y \
        ${PYTHON} \
        ${PYTHON}-dev \
        ${PYTHON}-tk \
        ${PYTHON}-pip

    sudo ${PIP} install --upgrade \
        pip \
        setuptools

    # Some TF tools expect a "python" binary
    sudo ln -s $(which python) /usr/local/bin/python

    sudo ${PIP} install \
#        tensorflow-gpu==2.0.0-beta1 \
        tensorflow-gpu==2.0.0-rc0 \
        tensorflow_probability==0.8.0rc0

    sudo apt update

    sudo -H ${PIP} install --ignore-installed PyYAML

    # fixed numpy at 1.16 since 1.17 had incompatiblities with tf. Will be fixed in 1.18
    sudo ${PIP} install \
        apache-beam[gcp] \
        pillow==4.0.0 \
        numpy==1.16.4 \
        imageio \
        matplotlib \
        opencv-python

    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        git \
        nano \
        curl \
        htop \
        sysstat \
        unzip \
        firewalld

    sudo echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    sudo curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    sudo apt update
    sudo apt-get install tensorflow-model-server

    sudo apt-get update

    PROJECT_DIR="$HOME_DIR/pycharm_project"

    sudo mkdir $PROJECT_DIR
    cd $PROJECT_DIR

    sudo chown -R $GCLOUD_USERNAME $HOME_DIR
    sudo echo -e "cd $HOME_DIR" >> /etc/profile.d/logincommands.sh

    echo "Complete installation finished."
    NEEDS_REBOOT='true'

else

    echo "CUDA already installed"

fi


if $NEEDS_REBOOT; then
    ## Reboot system to initialize changes
    echo "Rebooting system to initialize system changes."
    sudo reboot

else
    ###### Stuff that should be done on each boot. Note for Azure the gcloud_init code is only used on first boot this won't run
    echo "Enabling persistence mode for better performance"
    sudo nvidia-smi -pm 1

    # On instances with NVIDIA® Tesla® K80 GPUs, disable autoboost for better performance (https://cloud.google.com/compute/docs/gpus/add-gpus):
    if [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader) == "Tesla K80" ]]; then
        echo -e "GPU name is Tesla K80 --> Disabling auto-boost-default to improve performance."
        sudo nvidia-smi --auto-boost-default=DISABLED
    fi

fi
