import os
import cv2
import random

# Configuration
RAW_DIR = "Thumos14 Raw"
PROCESSED_DIR = "Thumos14 processed"
NUM_SEQUENCES_PER_VIDEO = 4
NUM_FRAMES_PER_SEQUENCE = 5
TARGET_SIZE = (640, 360)  # (width, height)

# Keywords to exclude (videos with these substrings in the filename will be skipped)
EXCLUDE_KEYWORDS = ["ApplyEyeMakeup", "ApplyLipstick"]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_FRAMES_PER_SEQUENCE:
        print(f"Video {video_path} does not have enough frames ({total_frames})")
        cap.release()
        return

    max_start = total_frames - NUM_FRAMES_PER_SEQUENCE
    # Randomly choose starting frame indices for sequences
    if max_start + 1 >= NUM_SEQUENCES_PER_VIDEO:
        start_indices = random.sample(range(0, max_start + 1), NUM_SEQUENCES_PER_VIDEO)
    else:
        start_indices = list(range(0, max_start + 1))
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    for seq_idx, start in enumerate(start_indices):
        # Create a unique sequence folder directly under PROCESSED_DIR.
        seq_folder = os.path.join(PROCESSED_DIR, f"{video_basename}_seq_{seq_idx+1:04d}")
        os.makedirs(seq_folder, exist_ok=True)
        
        # Set video to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for frame_idx in range(NUM_FRAMES_PER_SEQUENCE):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_path} at start {start}")
                break
            # Resize frame to target resolution (640x360)
            frame_resized = cv2.resize(frame, TARGET_SIZE)
            frame_filename = os.path.join(seq_folder, f"{frame_idx+1:04d}.jpg")
            cv2.imwrite(frame_filename, frame_resized)
        print(f"Saved sequence starting at frame {start} to {seq_folder}")
    
    cap.release()

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    video_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.avi')]
    print(f"Found {len(video_files)} video files in {RAW_DIR}")
    
    for video_file in video_files:
        # Exclude videos that have undesired keywords in the filename.
        if any(keyword in video_file for keyword in EXCLUDE_KEYWORDS):
            print(f"Skipping excluded video: {video_file}")
            continue
        
        video_path = os.path.join(RAW_DIR, video_file)
        process_video(video_path)

if __name__ == "__main__":
    main()
