import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##############################################
# Center Crop Module (TorchScript-friendly)
##############################################
class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """
        Args:
          crop_size (tuple): (new_height, new_width)
        """
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        new_H, new_W = self.crop_size
        start_H = (H - new_H) // 2
        start_W = (W - new_W) // 2
        return x[:, :, start_H:start_H+new_H, start_W:start_W+new_W]

##############################################
# Helper Modules
##############################################
class deconv(nn.Module):
    """
    Upsamples using bilinear interpolation and then applies a convolution.
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0, scale_factor=2):
        super(deconv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return self.conv(x)

class HierarchyEncoder(nn.Module):
    """
    Hierarchical encoder with fixed residual connections.
    Splits channels into groups and concatenates the original input with intermediate outputs.
    """
    def __init__(self, channel):
        super(HierarchyEncoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        out = x
        x_cont = x.contiguous()
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and i != 0:
                g = self.group[i // 2]
                x0 = x_cont.view(bt, g, -1, h, w)
                out0 = out.view(bt, g, -1, h, w)
                out = torch.cat([x0, out0], dim=2).view(bt, -1, h, w)
            out = layer(out)
        return out

class Vec2PatchAlternative(nn.Module):
    """
    Reconstructs a patch image from token features.
    
    Args:
      channel: desired output channels (e.g., 128)
      hidden: hidden dimension (e.g., 512)
      enc_size: spatial size of the encoder's output (here, (90, 160))
      output_size: target patch image size (should match enc_size, here (90, 160))
      kernel_size, stride, padding: parameters used in patch2vec (e.g., (7,7), (3,3), (3,3))
    """
    def __init__(self, channel, hidden, enc_size, output_size, kernel_size, stride, padding):
        super(Vec2PatchAlternative, self).__init__()
        self.c_out = kernel_size[0] * kernel_size[1] * channel  # e.g., 7*7*128 = 6272
        self.embedding = nn.Linear(hidden, self.c_out)
        self.output_size = output_size  # (90, 160)
        self.H, self.W = output_size

        # Compute the number of tokens produced by patch2vec.
        H_enc, W_enc = enc_size  # (120,214)
        H_out = (H_enc + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        W_out = (W_enc + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        self.num_tokens = H_out * W_out  # Expect: 30 * 54 = 1620 tokens

        total_elements = self.num_tokens * self.c_out
        self.new_channels = total_elements // (self.H * self.W)
        self.reduce_conv = nn.Conv2d(self.new_channels, channel, kernel_size=1)

    def forward(self, x):
        # x: (B, num_tokens, hidden)
        B = x.size(0)
        feat = self.embedding(x)
        feat = feat.transpose(1, 2)
        feat = feat.contiguous().view(B, -1)
        target = self.new_channels * self.H * self.W
        feat = feat[:, :target]
        feat = feat.view(B, self.new_channels, self.H, self.W)
        feat = self.reduce_conv(feat)
        return feat

##############################################
# Transformer Blocks: Spatial and Temporal with Memory
##############################################
class SpatialTransformerBlockWithMemory(nn.Module):
    """
    Spatial block: applies standard multi-head attention to the current frame tokens
    concatenated with memory tokens.
    
    Inputs:
      x: (B, n, hidden) current frame tokens.
      memory: (memory_slots, n, hidden) tokens from previous frames.
    Outputs:
      x_final: (B, n, hidden) processed current frame tokens.
      new_memory: updated memory (memory_slots, n, hidden).
    """
    def __init__(self, hidden, num_head, dropout=0.1):
        super(SpatialTransformerBlockWithMemory, self).__init__()
        self.hidden = hidden
        self.num_head = num_head
        self.head_dim = hidden // num_head
        self.query_linear = nn.Linear(hidden, hidden)
        self.key_linear   = nn.Linear(hidden, hidden)
        self.value_linear = nn.Linear(hidden, hidden)
        self.out_linear   = nn.Linear(hidden, hidden)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden*4, hidden),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory):
        # x: (B, n, hidden); memory: (memory_slots, n, hidden).
        B, n, hidden = x.size()
        load = memory.view(-1, n, hidden)  # (memory_slots, n, hidden)
        x_cat = torch.cat([x, load], dim=0)  # ((B + memory_slots), n, hidden)
        t = x_cat.size(0)
        
        x_norm = self.norm1(x_cat)
        Q = self.query_linear(x_norm)
        K = self.key_linear(x_norm)
        V = self.value_linear(x_norm)
        
        Q = Q.view(t, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(t, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(t, n, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs  = F.softmax(attn_scores, dim=-1)
        attn_probs  = self.attention_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(t, n, self.hidden)
        attn_output = self.out_linear(attn_output)
        
        x_updated = x_cat + self.dropout(attn_output)
        y = self.norm2(x_updated)
        x_all = x_updated + self.ffn(y)
        
        x_final = x_all[:B]  # current frame tokens
        new_memory = torch.cat([memory[1:], x], dim=0)
        return x_final, new_memory

class TemporalTransformerBlockWithMemory(nn.Module):
    """
    Temporal block: replicates the original DSTT "t-mode" behavior using memory.
    It assumes x is (B, n, hidden) with n = h * w, where h and w are even.
    It concatenates memory with x along a temporal dimension, reshapes to 
      (t, 2, h//2, 2, w//2, num_head, head_dim),
    permutes to (t, num_groups, num_head, L, head_dim) where L = (h//2)*(w//2),
    then for each group uses the current frame’s tokens as query (Q) and the 
    corresponding tokens across time (memory + current frame) as keys (K) and values (V),
    computes scaled dot-product attention, and then reassembles the output.
    
    Memory is updated by dropping the oldest slot and appending the current frame tokens.
    """
    def __init__(self, hidden, num_head, h, w, dropout=0.1):
        super(TemporalTransformerBlockWithMemory, self).__init__()
        self.hidden = hidden
        self.num_head = num_head
        self.h = h  # e.g., 40
        self.w = w  # e.g., 72 (must be even)
        self.head_dim = hidden // num_head
        self.num_groups = 4  # splitting into 2x2 groups
        self.L = (self.h // 2) * (self.w // 2)  # tokens per group
        
        self.query_linear = nn.Linear(hidden, hidden)
        self.key_linear   = nn.Linear(hidden, hidden)
        self.value_linear = nn.Linear(hidden, hidden)
        self.out_linear   = nn.Linear(hidden, hidden)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden*4, hidden),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory):
        # x: (B, n, hidden) with n = h*w; memory: (memory_slots, n, hidden).
        B, n, hidden = x.size()
        t = B + memory.size(0)  # total time steps (current frame + memory)
        load = memory.view(-1, n, hidden)  # (memory_slots, n, hidden)
        x_cat = torch.cat([x, load], dim=0)  # (t, n, hidden)
        
        # Reshape x_cat to have spatial groups
        # Expected n = 4 * (h//2) * (w//2)
        x_cat = x_cat.view(t, 2, self.h // 2, 2, self.w // 2, self.num_head, self.head_dim)
        # Permute to (t, num_groups, num_head, L, head_dim) where num_groups = 2*2 = 4 and L = (h//2)*(w//2)
        x_cat = x_cat.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x_cat = x_cat.view(t, self.num_groups, self.num_head, self.L, self.head_dim)
        
        attn_outputs = []
        # Process each group separately.
        for g in range(self.num_groups):
            # Q for current frame in group g: shape (num_head, L, head_dim)
            Q_g = x_cat[-1, g]  
            # K, V: all time steps for group g: shape (t, num_head, L, head_dim)
            K_g = x_cat[:, g]
            V_g = x_cat[:, g]
            
            # Expand Q to match the temporal dimension: now (t, num_head, L, head_dim)
            Q_g_exp = Q_g.unsqueeze(0).expand(t, -1, -1, -1)
            
            # Rearrange to (t, L, num_head, head_dim) for scaled_dot_product_attention
            Q_g_exp = Q_g_exp.permute(0, 2, 1, 3)
            K_g = K_g.permute(0, 2, 1, 3)
            V_g = V_g.permute(0, 2, 1, 3)
            
            # Compute attention over time for this group.
            # The attention is computed for each time step, but we will take the result
            # corresponding to the current frame (last time step).
            attn_out_g = F.scaled_dot_product_attention(Q_g_exp, K_g, V_g,
                                                        dropout_p=self.attention_dropout.p,
                                                        is_causal=False)
            # Select the current frame’s output: shape (L, num_head, head_dim)
            attn_current = attn_out_g[-1]
            attn_outputs.append(attn_current)
        
        # Stack results from all groups: shape (num_groups, L, num_head, head_dim)
        attn_output = torch.stack(attn_outputs, dim=0)
        # Rearrange to (num_groups, L, num_head, head_dim) then flatten head dims:
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(self.num_groups, self.L, self.hidden)
        
        # Average over groups to fuse them
        attn_output = attn_output.mean(dim=0, keepdim=True)  # (1, L, hidden)
        # Tile to recover full token count (n = num_groups * L)
        x_att = attn_output.repeat(1, self.num_groups, 1)  # (1, 4*L, hidden)
        
        # Add the attention result to the current frame tokens.
        # Note: current_frame here is assumed to be x (with B=1)
        current_frame = x
        x_updated = current_frame + self.dropout(x_att)
        y = self.norm2(x_updated)
        x_final = x_updated + self.ffn(y)
        
        # Update memory: drop oldest and add current frame.
        new_memory = torch.cat([memory[1:], current_frame], dim=0)
        return x_final, new_memory


##############################################
# Inpaint Generator with Memory (for 360x640 input)
##############################################
class ONNXInpaintGeneratorWithMemory(nn.Module):
    def __init__(self, hidden=512, num_transformer_blocks=8, num_head=4, memory_slots=2, dropout=0.0, patch_stride=(3,3)):
        """
        For input images of resolution 360x640.
        The encoder outputs a feature map, which we force to a fixed size of (90, 160) via center cropping.
        With patch_stride=(3,3) and enc_size=(90, 160), patch2vec produces output of size (B, hidden, 30, 54),
        so seq_len = 30 * 54 = 1620 tokens.
        Transformer blocks are alternated between spatial and temporal modes.
        For temporal blocks, we assume the token grid is 30 x 54.
        """
        super(ONNXInpaintGeneratorWithMemory, self).__init__()
        channel = 256
        kernel_size = (7, 7)
        stride = patch_stride  # (3,3)
        padding = (3, 3)
        # Set encoder output size to (90, 160)
        self.enc_output_size = (90, 160)
        self.center_crop = CenterCrop(self.enc_output_size)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 360→180, 640→320
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 180→90, 320→160
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hier_enc = HierarchyEncoder(channel)
        self.patch2vec = nn.Conv2d(channel // 2, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        # With enc_size = (90,160): 
        # H_out = floor((90+6-7)/3)+1 = 30, W_out = floor((160+6-7)/3)+1 = 54.
        enc_size = self.enc_output_size
        output_size = self.enc_output_size
        self.vec2patch = Vec2PatchAlternative(channel // 2, hidden, enc_size, output_size, kernel_size, stride, padding)
        
        self.num_blocks = num_transformer_blocks
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_transformer_blocks):
            if i % 2 == 0:
                self.transformer_blocks.append(SpatialTransformerBlockWithMemory(hidden, num_head, dropout))
            else:
                self.transformer_blocks.append(TemporalTransformerBlockWithMemory(hidden, num_head, h=30, w=54, dropout=dropout))
        
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )
        
        # Final crop to match the input resolution (360, 640).
        # The decoder currently outputs (360,640) so we crop 1 pixel from each side in width.
        self.final_crop = CenterCrop((360, 640))
        
    def forward(self, masked_frames, memory):
        """
        masked_frames: (B, 3, 360, 640)
        memory: (1, num_blocks, memory_slots, seq_len, hidden)
                Here, seq_len = 30 * 54 = 1620 tokens.
        """
        memory = memory.squeeze(0)
        enc_feat = self.encoder(masked_frames)    # e.g., (B, 256, 90, 160)
        enc_feat = self.center_crop(enc_feat)       # Crop to (B, 256, 90, 160)
        enc_feat = self.hier_enc(enc_feat)          # (B, 128, 90, 160)
        trans_feat = self.patch2vec(enc_feat)        # (B, hidden, 30, 54)
        b, c, h, w = trans_feat.size()              # h=30, w=54 → seq_len = 1620
        seq_len = h * w
        trans_feat = trans_feat.view(b, c, seq_len).permute(0, 2, 1)  # (B, 1620, hidden)
        
        new_memories = []
        x = trans_feat
        for i, block in enumerate(self.transformer_blocks):
            mem_i = memory[i]  # (memory_slots, seq_len, hidden)
            x, new_mem = block(x, mem_i)
            new_memories.append(new_mem)
        new_memory = torch.stack(new_memories, dim=0)
        
        trans_feat = self.vec2patch(x)  # (B, channel//2, 90, 160)
        enc_feat = enc_feat + trans_feat
        output = self.decoder(enc_feat)  # Expected output shape: (B, 3, 360, 640)
        output = torch.tanh(output)
        # Crop to match the input resolution (360, 640)
        output = self.final_crop(output)
        return output, new_memory

##############################################
# Main: Inference and Export
##############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    hidden = 512
    num_transformer_blocks = 8
    memory_slots = 2

    dummy_input = torch.randn(1, 3, 360, 640, device=device)
    seq_len = 30 * 54  # 1620 tokens
    dummy_memory = torch.randn(1, num_transformer_blocks, memory_slots, seq_len, hidden, device=device)
    
    model = ONNXInpaintGeneratorWithMemory(
        hidden=hidden,
        num_transformer_blocks=num_transformer_blocks,
        memory_slots=memory_slots,
        patch_stride=(3,3)
    ).to(device)
    checkpoint_path = "checkpoints/stage2_checkpoint2.pth"  # Update with your actual file path.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['model_state'])
    print("Loaded checkpoint from", checkpoint_path)

    model.eval()

    with torch.no_grad():
        output, new_memory = model(dummy_input, dummy_memory)
    
    scripted_model = torch.jit.script(model)
    print("Scripted model successfully created.")
    
    torch.onnx.export(
        scripted_model,
        (dummy_input, dummy_memory),
        "DSTT_OM_RT.onnx",
        input_names=['masked_frames', 'memory'],
        output_names=['output', 'new_memory'],
        opset_version=18,
        dynamic_axes={
            'masked_frames': {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'memory': {0: 'num_blocks'},
            'new_memory': {0: 'num_blocks'}
        }
    )
    print("Model with alternating transformer blocks successfully exported to DSTT_OM_RT.onnx")
