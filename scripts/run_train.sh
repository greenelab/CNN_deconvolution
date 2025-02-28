#!/bin/sh

#SBATCH --job-name=train_MLP_MNIST
#SBATCH --account=amc-general
#SBATCH --output=output_train_MLP_MNIST.log
#SBATCH --error=error_train_MLP_MNIST.log
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

# Activate the virtual environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_cnn

# Update PYTHONPATH
export PYTHONPATH="CNN_deconvolution"
echo "PYTHONPATH in SLURM: $PYTHONPATH"

# Define parameters
MODEL_TYPE="MLP"
DATASET="MNIST"
OUTPUT_PATH="CNN_deconvolution/results/"
CONFIG_PATH="CNN_deconvolution/data/images/"
PCAM_DATA_PATH=""

# Run the script with parameters
echo "Running script with model_type=$MODEL_TYPE, dataset=$DATASET..."
srun $(which python) "CNN_deconvolution/scripts/train.py \
    --model_type $MODEL_TYPE \
    --dataset $DATASET \
    --config_path $CONFIG_PATH \
    --output_path $OUTPUT_PATH \
    --pcam_data_path $PCAM_DATA_PATH

# Deactivate the conda environment
conda deactivate


