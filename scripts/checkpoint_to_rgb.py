import torch

# Load your 12-band teacher checkpoint
ckpt = torch.load("/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_full/eval/training_12499/teacher_checkpoint.pth", map_location="cpu")
state_dict = ckpt["teacher"] # or "model" depending on your keys

# The patch embedding layer is the only one that changes with channel count
# Standard ViT name is 'backbone.patch_embed.proj.weight'
weight_key = "backbone.patch_embed.proj.weight" 
original_weight = state_dict[weight_key] # Shape: [embed_dim, 12, 16, 16]

# Slice to get only RGB (assuming your 12-band order starts with RGB)
# If your RGB bands were at specific indices, e.g., [3, 2, 1], use: 
# state_dict[weight_key] = original_weight[:, [3, 2, 1], :, :]
state_dict[weight_key] = original_weight[:, :3, :, :] 

# Save as a new '3-band version' of your model
torch.save({"teacher": state_dict}, "dinov3-12b_rgb.pth")