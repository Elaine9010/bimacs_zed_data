import os
import cv2
import argparse

def play_demo(input_folder):
    
    demo_name = input_folder.split('zed_raw_data/')[1].split('/rgb_pngs')[0]
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))])
    frame_count = len(frame_files)
    if frame_count == 0:
        print("No frames found in the input folder.")
        return
    
    current_frame = 0

    while True:
        # Load and resize the current frame
        frame_path = os.path.join(input_folder, frame_files[current_frame])
        frame = cv2.imread(frame_path)
        frame_resized = cv2.resize(frame, (960, 560))

        # Display the frame number on the frame
        frame_number_text = f"Frame: {current_frame}/{frame_count-1}"
        cv2.putText(frame_resized, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow(f'Demo: {demo_name}', frame_resized)
        key = cv2.waitKey(0)

        # Navigation
        if key == ord('d') and current_frame < frame_count - 1:
            current_frame += 1
        elif key == ord('a') and current_frame > 0:
            current_frame -= 1

        # Exit
        elif key == 27:  # Escape key
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()

#Usage: python play_demo.py --input ./zed_raw_data/subject_3/task_1_k_cooking/take_0/rgb_pngs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Label dataset frames.")
    parser.add_argument("--input", help="Path to the input folder containing the frames.")
    args = parser.parse_args()

    play_demo(args.input)
