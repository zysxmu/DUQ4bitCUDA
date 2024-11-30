#!/bin/bash

if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set! Simply set to /usr/local/cuda-12.1"
    export CUDA_HOME=/usr/local/cuda-12.1
else
    echo "CUDA_HOME is set to $CUDA_HOME."
fi

CONDA_PATH=$(dirname $(dirname $(which conda)))
source "${CONDA_PATH}/etc/profile.d/conda.sh"
echo "Setting up environment"

git submodule update --init --recursive

# conda environment setup
if conda info --envs | grep -qw ERQ; then
    echo "ERQ Environment already exists!"
else
    echo "Creating ERQ Environment..."
    conda create --name ERQ python=3.9 -y
fi
conda activate ERQ

echo "Installing dependencies"
echo "Installing pytorch"
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
echo "Installing cudatoolkit"
conda install -c nvidia cuda-nvcc==12.1.66 -y
conda install -c anaconda cmake==3.18.2 -y
export PATH=$(dirname $(dirname $(which conda)))/envs/ERQ/bin:$PATH
pip install -r requirements.txt

echo "====================================================="
echo "||              Environment details                ||"
echo "====================================================="
which cmake && cmake --version
which make && make --version
which gcc && gcc --version
which nvcc && nvcc --version
which python && python --version
echo "====================================================="

cd lib && pip install -e . --verbose && cd ..

echo "Environment setup complete!"
