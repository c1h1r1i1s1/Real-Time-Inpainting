import os
import requests
from bs4 import BeautifulSoup

# Base URL for the SUN3D data directory.
BASE_URL = "https://sun3d.cs.princeton.edu/data/"

# Folders to exclude.
EXCLUDED_DIRS = {
    "APCdata", "Portland_hotel", "SUNRGBD", "SUNRGBDv2", "SUNRGBDv2Test",
    "align_kv2", "rawdata_beforecrop", "demo", "kinect2data"
}

# Local folder (base directory) for saving downloaded sequences.
LOCAL_SAVE_DIR = "./SUN3D_frames"

# Output file for aria2 input.
OUTPUT_FILE = "aria2_sun3d_input.txt"

# Sampling parameters:
NUM_FRAMES = 5    # each group consists of 5 frames
FRAME_INTERVAL = 2  # take every 2nd frame (i, i+2, i+4, i+6, i+8)
SKIP_FRAMES = 40  # then skip 40 frames before next group
# Total frames used per group = NUM_FRAMES * FRAME_INTERVAL = 10 - 2? 
# Actually, for indices: i, i+2, i+4, i+6, i+8, the total span is 8 frames,
# and then we skip 40, so the next group starts at i + 48.

def get_links(url, file_ext=None):
    """
    Retrieves a list of links from a given URL.
    If file_ext is provided, only returns links ending with that extension (e.g. '.jpg').
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

def main():
    with open(OUTPUT_FILE, "w") as fout:
        # Get the top-level directories from BASE_URL.
        top_dirs = get_links(BASE_URL)
        allowed_top_dirs = [d for d in top_dirs if d not in EXCLUDED_DIRS]
        print("Allowed top directories:", allowed_top_dirs)
        
        group_count = 0
        # For each allowed top directory...
        for top_dir in allowed_top_dirs:
            top_dir_url = os.path.join(BASE_URL, top_dir) + "/"
            # Get scan directories inside the top-level folder.
            scan_dirs = get_links(top_dir_url)
            for scan in scan_dirs:
                # Construct the URL to the "image" folder within each scan.
                image_dir_url = os.path.join(top_dir_url, scan, "image") + "/"
                print(f"Processing {image_dir_url}")
                image_files = get_links(image_dir_url, file_ext=".jpg")
                image_files.sort()  # assumes chronological order
                total_files = len(image_files)
                print(f"  Found {total_files} jpg files")
                
                # Iterate over the list using our grouping pattern.
                i = 0
                while i + (NUM_FRAMES - 1) * FRAME_INTERVAL < total_files:
                    group_count += 1
                    # Define the output directory for this group.
                    group_dir = os.path.join(LOCAL_SAVE_DIR, f"{top_dir}_{scan}_grp_{group_count:04d}")
                    # Write aria2 options for each file in this group.
                    for j in range(NUM_FRAMES):
                        idx = i + j * FRAME_INTERVAL
                        url = os.path.join(image_dir_url, image_files[idx])
                        # Define output file name as frame_0001.jpg, etc.
                        out_name = f"frame_{j+1:04d}.jpg"
                        # Write the block for this file.
                        fout.write(url + "\n")
                        fout.write(f"  dir={group_dir}\n")
                        fout.write(f"  out={out_name}\n\n")
                    # Advance index by the group span: (last index used is i + (NUM_FRAMES-1)*FRAME_INTERVAL)
                    # Then skip SKIP_FRAMES additional frames.
                    i += (NUM_FRAMES - 1) * FRAME_INTERVAL + SKIP_FRAMES
                    print(f"  Group {group_count} processed, next group starting index {i}")
        print(f"Aria2 input file written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
