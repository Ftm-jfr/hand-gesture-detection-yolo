import cv2
import mediapipe as mp
import os
import shutil


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

class_map = {
    "One": 0,
    "Two": 1,
    "Three": 2,
    "Four": 3,
    "Five": 4,
    "Like": 5
}


images_output = "dataset/images/all"
labels_output = "dataset/labels/all"
os.makedirs(images_output, exist_ok=True)
os.makedirs(labels_output, exist_ok=True)

root_folder = "videos"

total_saved = 0

for class_name, class_id in class_map.items():
    frames_folder = os.path.join(root_folder, class_name, "frames")

    if not os.path.exists(frames_folder):
        print(f"Warning: Frame folder for class '{class_name}' not found → {frames_folder}")
        continue

    print(f"\n≫≫ Processing class: {class_name} (ID: {class_id})")
    saved_in_class = 0

    for img_file in os.listdir(frames_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(frames_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        h, w, _ = image.shape

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Compute bounding
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Add margin
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Normalize for YOLO format
            center_x = ((x_min + x_max) / 2) / w
            center_y = ((y_min + y_max) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            # Generate a unique image name
            new_img_name = f"{class_name}_{img_file}"
            new_img_path = os.path.join(images_output, new_img_name)

            # Copy original image
            shutil.copy(img_path, new_img_path)

            # Create YOLO label file
            label_path = os.path.join(
                labels_output,
                new_img_name.replace('.jpg', '.txt')
                            .replace('.jpeg', '.txt')
                            .replace('.png', '.txt')
            )

            with open(label_path, 'w') as f:
                f.write(
                    f"{class_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

            saved_in_class += 1
            total_saved += 1

    print(f"   ✓ Class '{class_name}': {saved_in_class} images labeled and saved.")

print("\nDone!")
print(f"Total images processed and labeled: {total_saved}")
print(f"Images directory → {images_output}")
print(f"Labels directory → {labels_output}")
