# Bimanual Human Demonstration Dataset Processing

This repository contains scripts to process raw data recorded with a ZED camera. The raw dataset consists of both depth and RGB images. The repository includes tools to segment masks, generate 3d bounding boxes and spatial relations between objects in the scene.

## Requirements

Before running the scripts, ensure that you have the following dependencies installed:

- Python (version 3.11.9)
- ZED SDK and Python API
- `sam2` (Segment Anything Model) for generating segmentation masks

## Dataset Processing
### 1. Convert SVO to Images
The first step involves converting ZED .svo files into individual images for further processing.

### 2. Reorganize Data and Convert Images
After converting the raw SVO files to images, the next step is to reorganize the folder structure, separate depth and RGB images, and convert PNG images to JPG format for segmentation.

`python reorganize.py --input ./zed_raw_data/subject_1/task_1_k_cooking/take_0`

### 3. Generate Segmentation Masks
Use the Segment Anything Model (SAM2) to generate segmentation masks from the RGB images

### 4. Create 3D Objects from Depth Images
After generating the segmentation masks, the depth images can be used to create 3D object data.

`python create_3d_objects.py -i ./zed_raw_data/subject_1/task_1_k_cooking/take_0 -o ./zed_derived_data -v -s`

This script generates a dataset of 3D objects from the depth images and stores them in the output folder. The -v flag enables verbose output, and the -s flag enables additional settings like scaling.

### 5. Generate Spatial Relations
Finally, the spatial relationships between objects can be computed and saved to JSON files.

`python generate_relations.py -input_folder ./zed_derived_data/subject_1/task_1_k_cooking -output_folder ./zed_derived_data/subject_1/task_1_k_cooking`

This script processes the 3D object dataset and outputs JSON files describing the spatial relations between objects.
