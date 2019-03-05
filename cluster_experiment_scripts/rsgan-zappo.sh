#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
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
export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets
cp -r ~/zappos-shoes/zappos-dataset ${DATASET_DIR}/zappo
echo done
ls ${DATASET_DIR}/zappo
# Activate the relevant virtual environment:

source /home/s1832582/miniconda3/bin/activate mlp
cd ..
#python copy_images.py --data ~/mlpractical/cluster_experiment_scripts/train_info.csv --output ${DATASET_DIR}/ --image_loc ~/paintings/train --style Impressionism
python gan_loss_iter.py --image_size 128 --n_iter 51000 --input_folder ${DATASET_DIR}/zappo --output_folder ~/paintings_rsgan --n_gpu 4 --loss_D 1 --spectral True --gen_every 50000 --extra_folder ~/paintings_rsgan/zappo


