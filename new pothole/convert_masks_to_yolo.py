import os
import cv2
import numpy as np

def convert_masks_to_yolo(mask_dir, rgb_dir, yolo_label_dir):
    os.makedirs(yolo_label_dir, exist_ok=True)
    default_class_id = 0
    for mask_name in os.listdir(mask_dir):
        if not mask_name.endswith('.png'):
            continue
        mask_path = os.path.join(mask_dir, mask_name)
        rgb_path = os.path.join(rgb_dir, mask_name)
        if not os.path.exists(rgb_path):
            print(f"No RGB image for {mask_name}, skipping.")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_lines = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            yolo_lines.append(f"{default_class_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
        label_txt = os.path.join(yolo_label_dir, mask_name.replace('.png', '.txt'))
        with open(label_txt, 'w') as f:
            f.write('\n'.join(yolo_lines))
        print(f"Converted {mask_name} -> {label_txt}")

# Convert all sets
convert_masks_to_yolo('pothole600/pothole600/training/label', 'pothole600/pothole600/training/rgb', 'pothole600/pothole600/training/labels_yolo')
convert_masks_to_yolo('pothole600/pothole600/validation/label', 'pothole600/pothole600/validation/rgb', 'pothole600/pothole600/validation/labels_yolo')
convert_masks_to_yolo('pothole600/pothole600/testing/label', 'pothole600/pothole600/testing/rgb', 'pothole600/pothole600/testing/labels_yolo')

print("All conversions complete.") 