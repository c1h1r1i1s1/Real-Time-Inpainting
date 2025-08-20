import os
import cv2

# Configuration
RAW_DIR = "GTA-IM Raw"              # Folder containing dated subfolders
PROCESSED_DIR = "GTA-IM Processed"    # Folder where sequences will be saved
TARGET_SIZE = (640, 360)              # (width, height) for resizing
NUM_FRAMES = 5                      # Number of frames per sequence

def process_date_folder(date_folder_path, date_folder_name):
    """
    Process one dated folder from the GTA-IM Raw data.
    It extracts all .jpg files, sorts them, and partitions them into non-overlapping groups of 5.
    Each group is saved as its own sequence in the PROCESSED_DIR.
    """
    # List all files in the folder and filter for .jpg files only.
    files = os.listdir(date_folder_path)
    jpg_files = [f for f in files if f.lower().endswith(".jpg")]
    jpg_files.sort()  # Assumes that the filenames sort in chronological order.
    
    num_sequences = len(jpg_files) // NUM_FRAMES
    print(f"Processing '{date_folder_name}': {len(jpg_files)} jpg files, {num_sequences} full sequences")
    
    for seq_idx in range(num_sequences):
        # Extract a non-overlapping 5-frame sequence.
        sequence_files = jpg_files[seq_idx * NUM_FRAMES : seq_idx * NUM_FRAMES + NUM_FRAMES]
        
        # Create a unique folder name for this sequence.
        sequence_folder_name = f"{date_folder_name}_seq_{seq_idx+1:04d}"
        sequence_folder_path = os.path.join(PROCESSED_DIR, sequence_folder_name)
        os.makedirs(sequence_folder_path, exist_ok=True)
        
        # Process and save each frame in the sequence.
        for frame_idx, filename in enumerate(sequence_files):
            file_path = os.path.join(date_folder_path, filename)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not read image '{file_path}'")
                continue
            # Resize the image to the target resolution.
            resized = cv2.resize(img, TARGET_SIZE)
            output_filename = f"{frame_idx+1:04d}.jpg"  # e.g., 0001.jpg, 0002.jpg, etc.
            output_path = os.path.join(sequence_folder_path, output_filename)
            cv2.imwrite(output_path, resized)
        print(f"Saved sequence '{sequence_folder_name}'")

def main():
    # Create the processed directory if it doesn't exist.
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    # Get a sorted list of all dated subfolders in RAW_DIR.
    date_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]
    date_folders.sort()
    
    # Process each dated folder.
    for date_folder in date_folders:
        date_folder_path = os.path.join(RAW_DIR, date_folder)
        process_date_folder(date_folder_path, date_folder)

if __name__ == "__main__":
    main()
