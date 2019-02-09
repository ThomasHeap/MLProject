#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=s1832582

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/s1832582

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets
# Activate the relevant virtual environment:

source /home/s1832582/miniconda3/bin/activate mlp
cd ..
python copy_images.py --data ~/mlpractical/cluster_experiments_scripts/train_info.csv --output ${DATASET_DIR}/paintings --image_loc ~/paintings --genre abstract
python dcgan.py --dataset folder --dataroot ${DATASET_DIR}/paintings --cuda --outf ~/paintings_dcgan

