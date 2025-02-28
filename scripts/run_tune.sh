#!/bin/sh

#SBATCH --job-name=tune_CNN_MNIST
#SBATCH --account=amc-general
#SBATCH --output=output_CNN_MNIST.log
#SBATCH --error=error_tune_CNN_MNIST.log
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1              # <--- Only one rank
#SBATCH --cpus-per-task=64      # <--- The one rank has 64 CPUs
#SBATCH --exclude=c3cpu-c15-u34-2

''' The following file is meant to tune the model of choosing using tune.py '''
''' Please change variables below to match the model type and data type you want to tune '''

# Activate the virtual environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_cnn

ray stop

sleep 10
# Double-check which python and ray we are using:
which python
which ray

##########################################################################################

# Start Ray
# Pick a port
PORT=1190

export RAY_TMPDIR=""

# # Start Ray HEAD on this node:
# ray start --head --port=$PORT --temp-dir="$RAY_TMPDIR"

# Define parameters
MODEL_TYPE="CNN"
DATASET="MNIST"
TMP_DIR=""
WORKING_DIR=""
OUTPUT_PATH="CNN_deconvolution/data/images/best_configs_${MODEL_TYPE}_${DATASET}.json"
PCAM_DATA_PATH="CNN_deconvolution/data/pcamv1/"
NUM_ITERATIONS="1"

# Run the script with parameters
echo "Running script with model_type=$MODEL_TYPE, dataset=$DATASET..."
srun $(which python) CNN_deconvolution/scripts/tune.py \
    --model_type $MODEL_TYPE \
    --dataset $DATASET \
    --tmp_dir $TMP_DIR \
    --working_dir $WORKING_DIR \
    --output_path $OUTPUT_PATH \
    --pcam_data_path $PCAM_DATA_PATH \
    --num_iterations $NUM_ITERATIONS \

#############################################################################################
# Deactivate the conda environment and stop final ray
ray stop
sleep 5
conda deactivate
