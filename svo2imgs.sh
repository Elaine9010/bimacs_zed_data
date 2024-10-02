#!/bin/bash

usage() {
    echo "Usage: bash svo2imgs.sh <subject_number> <take_number>"
    echo "  <subject_number>: The number of the subject (e.g., 1, 2, 3)"
    echo "  <take_number>: The number of the take (e.g., 1, 2, 3)"
}

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    usage
    exit 1
fi

# Set parameters
SUB=$1
TAKE=$2

echo "Processing data for Subject $SUB, Take $TAKE"

# Convert svo to images
echo "Converting SVO to images..."
cd ~/han_project/zed-sdk/recording/export/svo/python
mkdir -p /zed_raw_data/subject_${SUB}/task_1_k_cooking/take_${TAKE}
python svo_export.py --mode 4 --input_svo_file /zed_raw_data/subject_${SUB}/task_1_k_cooking/take_${TAKE}.svo2 --output_path_dir /zed_raw_data/subject_${SUB}/task_1_k_cooking/take_${TAKE}
echo "SVO conversion completed."

# Reorganize folders, separate depth and rgb images, convert png imgs to jpgs and save them seperately for segmentation
echo "Reorganizing folders..."
cd /home/han/Documents/ZED
python reorganize.py --input /zed_raw_data/subject_${SUB}/task_1_k_cooking/take_${TAKE}

echo "All processing tasks completed successfully for Subject $SUB, Take $TAKE."
