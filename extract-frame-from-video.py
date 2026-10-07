import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=5):
    os.makedirs(output_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Done: {video_name} -> {saved_count} frames saved.")


root_folder = 'videos'

if os.path.exists(root_folder):
    subfolders = [
        f for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f))
    ]

    for subfolder in subfolders:
        current_subfolder_path = os.path.join(root_folder, subfolder)
        output_frames_path = os.path.join(current_subfolder_path, 'frames')

        for file in os.listdir(current_subfolder_path):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_full_path = os.path.join(current_subfolder_path, file)
                print(f"Processing: {video_full_path} ...")
                extract_frames(video_full_path, output_frames_path)
else:
    print("Folder 'videos' not found.")
