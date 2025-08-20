import os
import cv2
import numpy as np
import random

# --- Configuration ---
ALLOWED_FOLDERS = [
    "office", "nursery", "livingroom", "kitchen",
    "garage", "gameroom", "dining-room", "corridor",
    "computerroom", "closet", "bedroom", "bathroom"
]
RAW_DIR = "MIT Indoor Scenes Raw"
PROCESSED_DIR = "MIT Indoor Scenes Processed"
TARGET_SIZE = (640, 360)  # (width, height)
NUM_FRAMES = 5  # Each still image becomes a 5-frame clip

# Parameters for minor transformations:
ROTATION_RANGE = (-2, 2)       # degrees
TRANSLATION_RANGE = (-10, 10)  # pixels in both x and y directions
BRIGHTNESS_RANGE = (0.95, 1.05)  # multiplicative factor

def apply_random_transform(image):
    """
    Apply a slight random affine transformation to the input image.
    This includes a small rotation, translation, and brightness adjustment.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Random rotation
    angle = random.uniform(*ROTATION_RANGE)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Random translation
    dx = random.uniform(*TRANSLATION_RANGE)
    dy = random.uniform(*TRANSLATION_RANGE)
    M[0, 2] += dx
    M[1, 2] += dy
    
    # Apply the affine transformation
    transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness adjustment
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    transformed = np.clip(transformed * brightness_factor, 0, 255).astype(np.uint8)
    
    return transformed

def process_image(image_path, output_seq_dir):
    """
    For the given image, create a 5-frame sequence with slight random transformations.
    The first frame is the resized image, subsequent frames are slight variations.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        return

    # Resize to target resolution
    image_resized = cv2.resize(image, TARGET_SIZE)
    
    # Create output directory for the sequence if it doesn't exist
    os.makedirs(output_seq_dir, exist_ok=True)
    
    # Save NUM_FRAMES frames: first one is the resized image,
    # and then apply minor transformations for each subsequent frame.
    for i in range(NUM_FRAMES):
        if i == 0:
            frame = image_resized
        else:
            frame = apply_random_transform(image_resized)
        frame_name = f"{i+1:04d}.jpg"  # e.g. 0001.jpg, 0002.jpg, ...
        frame_path = os.path.join(output_seq_dir, frame_name)
        cv2.imwrite(frame_path, frame)
    print(f"Processed sequence saved to {output_seq_dir}")

def main():
    # Create the main processed directory
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Loop over the allowed category folders in the raw data directory
    for category in ALLOWED_FOLDERS:
        raw_category_dir = os.path.join(RAW_DIR, category)
        if not os.path.isdir(raw_category_dir):
            print(f"Skipping missing folder: {raw_category_dir}")
            continue

        processed_category_dir = os.path.join(PROCESSED_DIR, category)
        os.makedirs(processed_category_dir, exist_ok=True)

        # List image files (you can add more extensions if needed)
        image_files = [f for f in os.listdir(raw_category_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        image_files.sort()

        # Process each image: each image becomes its own sequence
        for img_file in image_files:
            base_name, _ = os.path.splitext(img_file)
            # Create a unique sequence directory for each image (using the image's base name)
            output_seq_dir = os.path.join(processed_category_dir, base_name)
            image_path = os.path.join(raw_category_dir, img_file)
            process_image(image_path, output_seq_dir)

if __name__ == "__main__":
    main()
