import os
import tifffile as tiff
import torch
import torch.nn as nn

class ViTBinaryClassifier(nn.Module):
    def __init__(self, img_size_x, img_size_y, in_channels=1, num_classes=2, patch_size_x=16, patch_size_y=16, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072):
        super(ViTBinaryClassifier, self).__init__()
        
        # Embedding: Patch + Positional Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y))
        
        num_patches = (img_size_x // patch_size_x) * (img_size_y // patch_size_y)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, activation='gelu'), 
            num_layers=depth
        )
        
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Input: (batch_size * patches_per_image, in_channels, patch_size_x, patch_size_y)
        x = self.patch_embed(x)  # (batch_size * patches_per_image, embed_dim, num_patches_x, num_patches_y)
        x = x.flatten(2).transpose(1, 2)  # (batch_size * patches_per_image, num_patches, embed_dim)
        
        # Add class token and positional embedding
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size * patches_per_image, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size * patches_per_image, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        
        # Transformer blocks
        x = self.transformer(x)  # (batch_size * patches_per_image, num_patches + 1, embed_dim)
        
        # Use the [CLS] token for classification
        cls_output = x[:, 0]  # (batch_size * patches_per_image, embed_dim)
        
        # Classification head
        out = self.fc(cls_output)  # (batch_size * patches_per_image, num_classes)
        return out