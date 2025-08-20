import os
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np

# Base URL for the SUN3D data directory
BASE_URL = "https://sun3d.cs.princeton.edu/data/"

# Directories we want to skip (everything before brown_bm_1)
    EXCLUDED_DIRS = {
        "APCdata", "Portland_hotel", "SUNRGBD", "SUNRGBDv2", "SUNRGBDv2Test",
        "align_kv2", "rawdata_beforecrop", "demo", "kinect2data"
    }

# Local folder to save downloaded sequences (each sequence in its own folder)
LOCAL_SAVE_DIR = "./SUN3D_frames"

# Number of frames per sequence and frame skip
NUM_FRAMES = 5        # Each sequence will have 5 frames.
FRAME_SKIP = 3        # Use every 3rd frame.

# Target resolution (width, height) that your model uses.
TARGET_SIZE = (640, 360)

def get_links(url, file_ext=None):
    """
    Retrieves a list of links from a given URL.
    If file_ext is provided, only returns links ending with that extension (e.g., '.jpg').
    Otherwise, returns links that look like directories (ending with a slash).
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href in ("../", "/", "/data/"):
            continue
        if file_ext:
            if href.lower().endswith(file_ext.lower()):
                links.append(href)
        else:
            if href.endswith("/"):
                links.append(href.strip("/"))
    return links

def download_and_process_image(url, save_path):
    """
    Downloads an image from the URL, resizes it to TARGET_SIZE, and saves it locally.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return
    
    data = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not decode image from {url}")
        return
    
    resized = cv2.resize(img, TARGET_SIZE)
    cv2.imwrite(save_path, resized)
    print(f"Saved {save_path}")

def main():
    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR)
    
    # Retrieve top-level directories.
    top_dirs = get_links(BASE_URL)
    allowed_dirs = [d for d in top_dirs if d not in EXCLUDED_DIRS]
    print("Allowed directories:", allowed_dirs)
    
    # Optionally, start from a specific directory:
    # start = False
    for top_dir in allowed_dirs:
        # if not start:
        #     if top_dir == "mit_lab_16":
        #         start = True
        #     else:
        #         continue
        top_dir_url = os.path.join(BASE_URL, top_dir) + "/"
        
        # Get the scan directories inside the top-level folder.
        scan_dirs = get_links(top_dir_url)
        for scan in scan_dirs:
            # Construct the URL to the image folder within each scan.
            image_dir_url = os.path.join(top_dir_url, scan, "image") + "/"
            print(f"Processing images in {image_dir_url}")
            
            # Retrieve and sort the list of .jpg files.
            image_files = get_links(image_dir_url, file_ext=".jpg")
            print(f"Found {len(image_files)} image files")
            image_files.sort()
            
            # Determine the number of non-overlapping sequences.
            seq_frame_count = NUM_FRAMES * FRAME_SKIP
            num_sequences = len(image_files) // seq_frame_count
            print(f"Extracting {num_sequences} sequences from this scan")
            
            for seq_idx in range(num_sequences):
                # Create a unique folder for the sequence directly in LOCAL_SAVE_DIR.
                # Folder name includes the top_dir and scan names.
                seq_folder = os.path.join(
                    LOCAL_SAVE_DIR, f"{top_dir}_{scan}_seq_{seq_idx+1:04d}"
                )
                os.makedirs(seq_folder, exist_ok=True)
                
                for j in range(NUM_FRAMES):
                    idx = seq_idx * seq_frame_count + j * FRAME_SKIP
                    if idx >= len(image_files):
                        break
                    img_file = image_files[idx]
                    img_url = os.path.join(image_dir_url, img_file)
                    local_img_path = os.path.join(seq_folder, img_file)
                    download_and_process_image(img_url, local_img_path)
                print(f"Saved sequence {seq_idx+1} to {seq_folder}")

if __name__ == "__main__":
    main()
