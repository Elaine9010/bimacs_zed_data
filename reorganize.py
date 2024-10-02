import os
from PIL import Image
import argparse

def main(args):

    folder_path = args.input

    depth_path = os.path.join(folder_path, 'depth_pngs')
    png_path = os.path.join(folder_path, 'rgb_pngs')
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)

    ## Get all files that start with 'depth' and end with '.png', and move them to the new folder /depth_pngs
    files = [f for f in os.listdir(folder_path) if f.startswith('depth') and f.endswith('.png')]
    for filename in files:
        os.rename(os.path.join(folder_path, filename), os.path.join(depth_path, filename))

    ## Get all files that start with 'left' and end with '.png', and move them to the new folder /rgb_pngs
    files = [f for f in os.listdir(folder_path) if f.startswith('left') and f.endswith('.png')]
    for filename in files:
        os.rename(os.path.join(folder_path, filename), os.path.join(png_path, filename))

    jpg_path = os.path.join(folder_path, 'rgb_jpgs')
    os.makedirs(jpg_path, exist_ok=True)

    #check wether the jpg folder is empty, if not, print a message and return
    if os.listdir(jpg_path):
        print("The output folder is not empty")
        return
    
    image_count = 0
    part_number = 1
    
    files = sorted([f for f in os.listdir(png_path)])

    for filename in files:
        img = Image.open(os.path.join(png_path, filename))
        rgb_img = img.convert('RGB')
        
        # Remove the '.png' extension and "left" and replace it with '.jpg'
        new_filename = os.path.splitext(filename)[0].replace('left', '') + '.jpg'
        
        # Create a new folder for every 200 images
        if image_count % 200 == 0:
            part_folder = os.path.join(jpg_path, f'rgb_jpgs_p{part_number}')
            if not os.path.exists(part_folder):
                os.makedirs(part_folder)
            part_number += 1
            
        # Save the image in the appropriate folder
        rgb_img.save(os.path.join(part_folder, new_filename), 'JPEG')
        print(f"images saved in {part_folder}")
        
        image_count += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the input folder')
    args = parser.parse_args()
    main(args)