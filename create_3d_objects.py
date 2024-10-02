"""
This script processes the depth images and masks to generate 3D bounding boxes for each object instance.
And save 3d information of detected objects of each frame to json files

"""

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import json
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def cluster_points(pcd, eps=0.05, min_points=10):
    """
    Cluster points using DBSCAN algorithm.
    
    :param pcd: o3d.geometry.PointCloud object
    :param eps: DBSCAN epsilon parameter
    :param min_points: DBSCAN min_samples parameter
    :return: Filtered points of the largest cluster
    """
    point_cloud = np.asarray(pcd.points)
    try:
        clusters = pcd.cluster_dbscan(eps=0.05, min_points=10)
        unique_labels = set(clusters)
        clustered_points = [point_cloud[np.array(clusters) == label] for label in unique_labels if label != -1]
        clusters = sorted(clustered_points, key=lambda x: len(x), reverse=True)
        filtered_points = clusters[0]  # Get the largest cluster
        return filtered_points
    
    except Exception as e:
        print(f"[ERROR] Clustering failed: {str(e)}")
        return None

def bbox2dict(bbox):
    """
    Convert a 3D bounding box to a dictionary.
    
    :param bbox: 3D bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
    :return: Dictionary of bounding box parameters
    """
    return {
        "x0": bbox[0], # min_x
        "x1": bbox[3], # max_x
        "y0": bbox[1], # min_y
        "y1": bbox[4], # max_y
        "z0": bbox[2], # min_z
        "z1": bbox[5], # max_z
    }

def gaussian_smooth_bbox(bbox_history, sigma=2.5):
    """
    Apply Gaussian smoothing to a history of bounding boxes.
    
    :param bbox_history: List of bounding boxes [min_x, min_y, min_z, max_x, max_y, max_z]
    :param sigma: Standard deviation for Gaussian kernel
    :return: Smoothed bounding box
    """
    bbox_array = np.array(bbox_history)
    smoothed_bbox = gaussian_filter1d(bbox_array, sigma=sigma, axis=0)
    return smoothed_bbox[-1]  # Return the most recent smoothed bounding box

def process_frame(depth_image, mask, camera_intrinsics, bbox_history,smooth_flag):
    """
    Process a single frame to get the 3D bounding box.
    :param depth_image: 2D numpy array of depth values
    :param mask: 2D numpy array of mask values
    :param camera_intrinsics: o3d.camera.PinholeCameraIntrinsic object
    :param bbox_history: List of past bounding boxes for this object
    :return: 3D bounding box parameters [center_x, center_y, center_z, size_x, size_y, size_z]
    """
    # Create a point cloud from the masked depth image
    masked_depth_image = np.where(mask == 0, 0, depth_image)
    masked_depth_image = o3d.geometry.Image(np.ascontiguousarray(masked_depth_image))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=masked_depth_image, intrinsic=camera_intrinsics)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    
    # Cluster points using DBSCAN algorithm.
    filtered_points = cluster_points(downsampled_pcd, eps=0.05, min_points=10) 
    if filtered_points is None:
        return None

    # Get the 3D bounding box from filtered points: [min_x, min_y, min_z, max_x, max_y, max_z]
    min_coords = np.min(filtered_points, axis=0)
    max_coords = np.max(filtered_points, axis=0)
    bbox_3d = np.concatenate([min_coords, max_coords])

    # Apply Gaussian smoothing to the bounding box
    if len(bbox_history) > 0:
        bbox_history.append(bbox_3d)
        if len(bbox_history) > 15:  # Keep a history of 15 frames (500ms at 30fps)
            bbox_history.pop(0)
        
        smoothed_bbox_3d = gaussian_smooth_bbox(bbox_history)
    else:
        bbox_history.append(bbox_3d)
        smoothed_bbox_3d = bbox_3d
    
    return smoothed_bbox_3d if smooth_flag else bbox_3d

def visualize_3d_bbox(rgb_image,bboxes,camera_intrinsics):
    """
    Visualizes the 3D bounding boxes projected onto a 2D image.
    Returns:
        rgb_image (np.array): The image with the projected 3D bounding box drawn on it.
    """
    #use different colors for different classes 
    colors = [
                (0, 255, 0),     # Green
                (255, 0, 0),     # Red
                (0, 0, 255),     # Blue
                (255, 255, 0),   # Yellow
                (255, 165, 0),   # Orange
                (128, 0, 128),   # Purple
                (0, 255, 255),   # Cyan
                (255, 192, 203), # Pink
                (165, 42, 42),   # Brown
                (0, 128, 0),     # Dark Green
                (75, 0, 130),    # Indigo
                (255, 20, 147),  # Deep Pink
                (0, 0, 0),       # Black
                (128, 128, 128),  # Gray
                (173, 216, 230), # Light Blue
                (255, 105, 180) # Hot Pink
            ]
    for obj in bboxes:
        x0, y0, z0, x1, y1, z1 = obj["bbox"]
        color_idx = obj["class_id"]
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1],
            [x1, y1, z1], [x0, y1, z1]
        ])
        corners_2d = np.zeros((8, 2))
        for i, corner in enumerate(corners):
            corner = np.array(corner)
            corner = np.dot(corner, camera_intrinsics.intrinsic_matrix.T)
            corners_2d[i] = corner[:2] / corner[2]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        for edge in edges:
            pt1 = tuple(map(int, corners_2d[edge[0]]))
            pt2 = tuple(map(int, corners_2d[edge[1]]))
            cv2.line(rgb_image, pt1, pt2, colors[color_idx], 2)
        #show class_id beside the bounding box
        cv2.putText(rgb_image, f"class_{color_idx}", (int(corners_2d[0][0]), int(corners_2d[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[color_idx], 2)

    return rgb_image

def main(depth_image_path, mask_path, output_path, visual_flag,save_flag,smooth_flag):
    
    # Camera intrinsic parameters
    fx, fy, cx, cy = 1065.54, 1065.5, 946.52, 554.568
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, fx, fy, cx, cy)
    obj_class_names = ["bowl","knife","screwdriver","cutting board","whisk","hammer","bottel","cup",
                    "banana","cereals","sponge","wood","saw","hard drive","left hand","right hand"]
    #rgb color mapping (fake data)
    color_mapping = {"bowl": [0,255,0], "knife": [255,0,0], "screwdriver": [0,0,255], "cutting board": [255,255,0],
                    "whisk": [255,165,0], "hammer": [128,0,128], "bottel": [255,192,203], "cup": [0,255,0],
                    "banana": [255,0,0], "cereals": [0,0,255], "sponge": [255,255,0], "wood": [255,165,0],
                    "saw": [128,0,128], "hard drive": [255,192,203], "left hand": [0,255,0], "right hand": [255,0,0]}

    past_positions = {} # store positions of objects in last frame
    bbox_histories = {}  # store bbox histories for each object

    image_names = sorted(os.listdir(depth_image_path))
    # Loop through depth images in the order of file names (frame number)
    for image_name in tqdm(image_names, desc="Processing images"):      # image_name: depth000000.png
        bboxes = [] #for visualization
        #print(f"Processing {image_name}")
        depth_raw = cv2.imread(os.path.join(depth_image_path, image_name), cv2.IMREAD_UNCHANGED)
        frame_num = int(image_name.split('.')[0].split('depth')[1])
        frame_data = []

        frame_masks = [mask_name for mask_name in os.listdir(mask_path) if mask_name.startswith(f"{frame_num:06d}")]
        # For each mask of current frame :{frame_num}_{mask_num}_{instance_num}.png
        for mask_name in frame_masks:   # mask_name: 000000_002_001.png
            #print(f"Processing {mask_name}")
            class_num = int(mask_name.split('.')[0].split('_')[1])
            instance_num = int(mask_name.split('.')[0].split('_')[2])
            mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_UNCHANGED)
            
            # Get or initialize bbox history for this object
            obj_key = (class_num, instance_num)
            if obj_key not in bbox_histories:
                bbox_histories[obj_key] = []

            bbox_3d = process_frame(depth_raw, mask, camera_intrinsics, bbox_histories[obj_key],smooth_flag)
            if bbox_3d is None:
                # if no object detected, append bbox of current object from history
                if obj_key in past_positions.keys():
                    bbox_3d = past_positions[obj_key]
                else: 
                    print(f"No object detected in {mask_name}")
                    continue

            #append bbox and class id to bboxes for visualization
            bboxes.append({"bbox":bbox_3d,"class_id":class_num})

            if save_flag:
                # Save the  detection results to frame_data 
                if obj_key in past_positions.keys():
                    past_bbox = past_positions[obj_key]
                else: 
                    #if no past bounding box exists, use the current one
                    past_bbox = bbox_3d
                
                obj_data ={
                    "bounding_box": bbox2dict(bbox_3d),
                    "certainty": 1, 
                    "class_index": class_num,
                    "class_name": obj_class_names[class_num],
                    "colour": color_mapping[obj_class_names[class_num]],
                    "instance_name": f"{obj_class_names[class_num]}_{instance_num}",
                    "past_bounding_box":bbox2dict(past_bbox),
                }
                frame_data.append(obj_data)
                # Save the bounding box to the history
                past_positions[obj_key] = bbox_3d

        if visual_flag:
            rgb_image = cv2.imread(os.path.join(depth_image_path.replace("depth_pngs", "rgb_pngs"), image_name.replace("depth", "left")), cv2.IMREAD_COLOR)
            visualization = visualize_3d_bbox(rgb_image, bboxes, camera_intrinsics)
            txt = f"Frame ID: {frame_num}"
            cv2.putText(visualization, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
            # Display the image in a GUI window
            cv2.imshow('3D Bounding Box Visualization', visualization)
            # Wait for user input to move to the next image
            #key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            key = cv2.waitKey(1)  # Wait for 1ms
            if key == ord('q'):
                break
        if save_flag:
            if frame_data == []:
                print(f"No object detected in frame_{frame_num}")
                continue
            # Save the frame data into json file, create the path if it doesn't exist
            output_file = os.path.join(output_path, f'frame_{frame_num}.json')
            os.makedirs(output_path, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(frame_data, f, indent=2)
            print(f"frame_{frame_num} saved to {output_path}")

        frame_num += 1



# usage: python create_3d_objects.py -i /home/han/Documents/ZED/subject_2/task_1_k_cooking/take_0 -o /home/han/graphBim/multimodal_task_graph_learning/zed_data/zed_derived_data -v

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create 3D objects from depth images and masks")
    parser.add_argument("--input_path", "-i", type=str, help="Path to depth images and masks")
    parser.add_argument("--output_path", "-o",type=str, help="Path to save 3D objects")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualization mode,not saving results", default=False)
    parser.add_argument("--save", "-s", action="store_true", help="Save the results to json files", default=False)
    parser.add_argument("--smooth", "-g", action="store_true", help="Apply Gaussian smoothing to bounding boxes", default=False)
    args = parser.parse_args()
    
    depth_image_path = os.path.join(args.input_path, "depth_pngs")
    mask_path = os.path.join(args.input_path, "masks") 
    output_path = os.path.join(args.output_path, args.input_path.split("zed_raw_data/")[1] ,"3d_objects")
    visual_flag = args.visualize
    save_flag = args.save
    smooth_flag = args.smooth

    main(depth_image_path, mask_path, output_path, visual_flag,save_flag,smooth_flag)
