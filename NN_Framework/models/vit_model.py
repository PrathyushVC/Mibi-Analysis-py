import torch
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, img_size_x, img_size_y, in_channels=1, num_classes=2, 
                 patch_size_x=16, patch_size_y=16, embed_dim=768, num_heads=12, 
                 depth=12, mlp_dim=3072, dropout_rate=0.1, weight_decay=1e-5):
        super(ViTClassifier, self).__init__()

        self.hparams = {
            'img_size_x': img_size_x,
            'img_size_y': img_size_y,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'patch_size_x': patch_size_x,
            'patch_size_y': patch_size_y,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'depth': depth,
            'mlp_dim': mlp_dim,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay
        }

        self.patch_embed = nn.Conv2d(in_channels, 
            embed_dim, kernel_size=(patch_size_x, patch_size_y), 
            stride=(patch_size_x, patch_size_y))

        num_patches = (img_size_x // patch_size_x) * (img_size_y // patch_size_y)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, 
            dim_feedforward=mlp_dim, dropout=dropout_rate, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        #Testing none zero initializations
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        # (batch_size, embed_dim, num_patches_x, num_patches_y)
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        # (batch_size, 1, embed_dim), expand to match batch size
        cls_token = self.cls_token.expand(x.size(0), -1, -1)

        
        x = torch.cat((cls_token, x), dim=1) + self.pos_embed
        x = self.dropout(x)

       
        x = self.transformer(x)

        
        cls_output = x[:, 0]

        
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)

        
        out = self.fc(cls_output)  # (batch_size, num_classes)
        return out