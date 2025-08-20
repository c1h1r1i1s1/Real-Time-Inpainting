import os
import cv2
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# Import your inpainting model (adjust path as needed).
from DSTT_OM_RT import ONNXInpaintGeneratorWithMemory

###############################
# Dataset and Mask Generation
###############################

class SequenceDataset(Dataset):
    """
    Loads 5-frame sequences from a root folder.
    Each subfolder is assumed to contain 5 frames.
    """
    def __init__(self, root_dir, sequence_length=5, transform=None):
        self.root_dir = "training_data/" + root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # List all subdirectories (each representing one sequence)
        self.seq_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                         if os.path.isdir(os.path.join(self.root_dir, d))]
        self.seq_dirs.sort()
        
    def __len__(self):
        return len(self.seq_dirs)
    
    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        # List JPEG images in the sequence directory
        img_files = [os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
                     if f.lower().endswith('.jpg')]
        img_files.sort()  # assumes filenames are in chronological order.
        if len(img_files) < self.sequence_length:
            raise ValueError(f"Sequence {seq_dir} has fewer than {self.sequence_length} frames")
        images = []
        for i in range(self.sequence_length):
            img = Image.open(img_files[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # Return tensor of shape (sequence_length, C, H, W)
        return torch.stack(images, dim=0)

def generate_freeform_mask(image_size, area_ratio_range=(0.05, 0.2), stroke_width_range=(10, 40)):
    """
    Generate an irregular free-form mask using random strokes.
    The final mask's area (proportion of pixels=1) is adjusted to be between the given area_ratio_range.
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Draw a random number of strokes.
    num_strokes = random.randint(1, 5)
    for _ in range(num_strokes):
        num_points = random.randint(3, 10)
        points = []
        for _ in range(num_points):
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            points.append((x, y))
        stroke_width = random.randint(*stroke_width_range)
        # Draw polyline and fill it.
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=stroke_width)
        cv2.fillPoly(mask, [pts], 1)
    
    # Adjust the mask area by dilation or erosion so that the mask covers between 5% and 20% of the image.
    current_ratio = np.sum(mask) / (H * W)
    target_ratio = random.uniform(*area_ratio_range)
    kernel = np.ones((5, 5), np.uint8)
    if current_ratio < target_ratio:
        # Dilate until the ratio is at least the target.
        while np.sum(mask) / (H * W) < target_ratio:
            mask = cv2.dilate(mask, kernel, iterations=1)
    elif current_ratio > target_ratio:
        # Erode until the ratio is at most the target.
        while np.sum(mask) / (H * W) > target_ratio:
            mask = cv2.erode(mask, kernel, iterations=1)
    
    # Convert mask to torch tensor.
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask

def generate_mask(image_size):
    """
    Generate a rectangular binary mask for an image of size (H, W) 
    where the masked (hole) area is between 5% and 20% of the image area.
    Returns a tensor of shape (H, W) with 1 for the hole and 0 for valid regions.
    """
    H, W = image_size
    area = H * W
    target_area = area * random.uniform(0.05, 0.2)  # 5% to 20%
    aspect_ratio = random.uniform(0.5, 2.0)
    h_rect = int(round((target_area / aspect_ratio) ** 0.5))
    w_rect = int(round((target_area * aspect_ratio) ** 0.5))
    h_rect = min(h_rect, H)
    w_rect = min(w_rect, W)
    top = random.randint(0, H - h_rect)
    left = random.randint(0, W - w_rect)
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[top:top+h_rect, left:left+w_rect] = 1.0
    return mask

def perturb_mask(mask, max_shift=5):
    """
    Perturb the mask by shifting it randomly in x and y by up to max_shift pixels.
    """
    H, W = mask.shape
    shift_y = random.randint(-max_shift, max_shift)
    shift_x = random.randint(-max_shift, max_shift)
    return torch.roll(mask, shifts=(shift_y, shift_x), dims=(0, 1))

def generate_mask_sequence(image_size, sequence_length=5):
    """
    Generate a sequence of masks.
    With 50% probability, use a stable mask; otherwise, use a slightly perturbed free-form mask.
    """
    # 50% chance to use a free-form mask.
    use_freeform = random.random() < 0.5
    if use_freeform:
        base_mask = generate_freeform_mask(image_size, area_ratio_range=(0.05, 0.2))
    else:
        # Fallback to the simple rectangular mask.
        base_mask = generate_mask(image_size)  # your original rectangular mask generator
    
    masks = []
    # For stable mask, use the same mask; otherwise, perturb it per frame.
    stable = random.random() < 0.5
    if stable:
        for _ in range(sequence_length):
            masks.append(base_mask.clone())
    else:
        for _ in range(sequence_length):
            masks.append(perturb_mask(base_mask, max_shift=5))
    return torch.stack(masks, dim=0)


###############################
# Training Setup and Curriculum
###############################

# Define image transform (scaling to [0,1] then normalizing to [-1,1]).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloader(root_folder, batch_size=2, shuffle=True):
    dataset = SequenceDataset(root_folder, sequence_length=5, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def get_curriculum_dataloaders(batch_size=2):
    # Stage 1: GTA-IM Processed and MIT Indoor Scenes Processed.
    stage1_folders = ["GTA-IM Processed", "MIT Indoor Scenes Processed"]
    datasets_stage1 = [SequenceDataset(folder, sequence_length=5, transform=transform)
                       for folder in stage1_folders]
    from torch.utils.data import ConcatDataset
    stage1_dataset = ConcatDataset(datasets_stage1)
    stage1_loader = DataLoader(stage1_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Stage 2: SUN3D_frames.
    # stage2_loader = get_dataloader("SUN3D_frames", batch_size=batch_size, shuffle=True)
    # # Stage 3: Thumos14 Processed.
    # stage3_loader = get_dataloader("Thumos14 Processed", batch_size=batch_size, shuffle=True)
    
    return stage1_loader#, stage2_loader, stage3_loader

def inpainting_loss(output, target, mask, hole_weight=1.0, valid_weight=1.0):
    """
    Compute the weighted L1 loss for the hole and valid regions.
    """
    hole_loss = torch.abs(output - target) * mask
    valid_loss = torch.abs(output - target) * (1 - mask)
    return hole_weight * hole_loss.mean() + valid_weight * valid_loss.mean()

def compute_psnr(output, target):
    mse = ((output - target) ** 2).mean().item()
    return 100 if mse == 0 else 10 * math.log10(1.0 / mse)

def train_model(model, optimizer, dataloader, device, epochs, stage_name):
    model.train()
    # Memory parameters: (should match your model's configuration)
    num_blocks = 8
    memory_slots = 2
    token_count = 1620
    hidden = 512

    scaler = torch.GradScaler("cuda")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # batch shape: (B, 5, C, H, W)
            batch = batch.to(device)
            B, S, C, H, W = batch.shape
            
            # Generate mask sequence for each sample.
            masks = []
            for _ in range(B):
                masks.append(generate_mask_sequence((H, W), sequence_length=S))
            masks = torch.stack(masks, dim=0).to(device)  # (B, 5, H, W)
            masks = masks.unsqueeze(2)  # (B, 5, 1, H, W)
            
            # Create masked input.
            masked_input = batch * (1 - masks)
            
            # Initialize memory for the sequence (shape: (1, num_blocks, memory_slots, token_count, hidden))
            memory = torch.zeros(1, num_blocks, memory_slots, token_count, hidden, device=device)
            
            optimizer.zero_grad()
            outputs = []
            with torch.autocast("cuda"):
                # Process frames sequentially while updating memory.
                for t in range(S):
                    frame_input = masked_input[:, t, :, :, :]  # (B, C, H, W)
                    out, new_memory = model(frame_input, memory)
                    outputs.append(out)
                    memory = new_memory.detach()  # Detach to avoid backpropagating through time excessively.
                outputs = torch.stack(outputs, dim=1)  # (B, 5, C, H, W)
                loss = inpainting_loss(outputs, batch, masks, hole_weight=1.0, valid_weight=1.0)
                # loss.backward()
                # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"{stage_name} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

###############################
# Main Training Script with Checkpoints
###############################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    os.makedirs("checkpoints", exist_ok=True)
    
    # Instantiate your model.
    model = ONNXInpaintGeneratorWithMemory(
        hidden=512,
        num_transformer_blocks=8,
        memory_slots=2,
        patch_stride=(3,3)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0., 0.99))
    
    # Create curriculum dataloaders.
    stage1_loader = get_curriculum_dataloaders(batch_size=2) #, stage2_loader, stage3_loader
    
    # Stage 1: Train on GTA-IM and MIT Indoor Scenes Processed for 20 epochs.
    print("Starting Stage 1: GTA-IM and MIT Indoor Scenes Processed")
    model = train_model(model, optimizer, stage1_loader, device, epochs=20, stage_name="Stage 1")
    stage1_checkpoint = os.path.join("checkpoints", "stage1_final.pth")
    torch.save(model.state_dict(), stage1_checkpoint)
    print(f"Checkpoint saved: {stage1_checkpoint}")
    
    # # Stage 2: Train on SUN3D_frames for 40 epochs.
    # print("Starting Stage 2: SUN3D_frames")
    # model = train_model(model, optimizer, stage2_loader, device, epochs=40, stage_name="Stage 2")
    # stage2_checkpoint = os.path.join("checkpoints", "stage2_final.pth")
    # torch.save(model.state_dict(), stage2_checkpoint)
    # print(f"Checkpoint saved: {stage2_checkpoint}")
    
    # # Stage 3: Train on Thumos14 Processed for 40 epochs.
    # print("Starting Stage 3: Thumos14 Processed")
    # model = train_model(model, optimizer, stage3_loader, device, epochs=40, stage_name="Stage 3")
    # stage3_checkpoint = os.path.join("checkpoints", "stage3_final.pth")
    # torch.save(model.state_dict(), stage3_checkpoint)
    # print(f"Checkpoint saved: {stage3_checkpoint}")
    
    # # Save final model.
    # torch.save(model.state_dict(), "inpainting_model.pth")
    # print("Training complete. Model saved as inpainting_model.pth")

if __name__ == "__main__":
    main()
