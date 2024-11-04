import torch
import torch.nn as nn

class SwinTransformer(nn.Module):
    """Swin Transformer implementation.

    A hierarchical vision transformer that uses shifted windows for efficient attention computation.
    Based on "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" 
    (https://arxiv.org/abs/2103.14030)

    Args:
        img_size (int): Input image size. Default: 224
        in_channels (int): Number of input channels. Default: 3
        patch_size (int): Patch size for patch embedding. Default: 16
        num_classes (int): Number of classes for classification. Default: 2
        embed_dim (int): Initial embedding dimension. Default: 96
        depths (List[int]): Number of layers in each stage. Default: [2, 2, 6, 2]
        num_heads (List[int]): Number of attention heads in each stage. Default: [3, 6, 12, 24]
        window_size (int): Window size for window attention. Default: 7
        mlp_ratio (float): MLP hidden dim expansion ratio. Default: 4.0
        dropout_rate (float): Dropout rate. Default: 0.1
        weight_decay (float): Weight decay factor. Default: 0.05

    Returns:
        torch.Tensor: Classification logits of shape (batch_size, num_classes)
    """


    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_classes=2,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0, dropout_rate=0.1, weight_decay=0.05):
        super().__init__()

        self.hparams = {
            'img_size': img_size,
            'patch_size': patch_size,
            'embed_dim': embed_dim,
            'depths': depths,
            'num_heads': num_heads,
            'window_size': window_size,
            'mlp_ratio': mlp_ratio,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay
        }

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim * 2 ** (self.num_layers - 1)
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Initialize current size based on patch embedding
        self.current_size = img_size // patch_size

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicSwinLayer(
                dim=embed_dim * 2**i_layer,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                downsample=(i_layer < self.num_layers - 1),
                size=self.current_size
            )
            self.layers.append(layer)
            if layer.downsample is not None:
                self.current_size = self.current_size // 2  # Update size after downsampling

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.num_features, num_classes)
        

    def forward(self, x):
        x = self.patch_embed(x)  # B, E, S, S
        #B, E, S, _ = x.shape
        #print(f"B:{B}",f"E:{E}",f"S:{S}")
        x = x.flatten(2).transpose(1, 2)  # B, S*S, E

        for layer in self.layers:
            x, self.current_size = layer(x, self.current_size)

        x = self.norm(x)
        x = x.transpose(1, 2)  # B, E, S*S
        x = self.avgpool(x)  # B, E, 1
        x = torch.flatten(x, 1)  # B, E
        x = self.fc(x)  # B, num_classes

        return x


class PatchEmbedding(nn.Module):
    """Splits image into patches and linearly embeds them."""
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, S, S
        x = x.flatten(2).transpose(1, 2)  # B, S*S, C
        x = self.norm(x)
        return x


class BasicSwinLayer(nn.Module):
    """Basic Swin Transformer layer.
    
    This layer implements a sequence of Swin Transformer blocks with optional downsampling.
    It alternates between regular window attention and shifted window attention in the blocks.
    
    Args:
        dim (int): Number of input channels/embedding dimension
        depth (int): Number of Swin blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Size of attention window
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0
        dropout (float, optional): Dropout rate. Defaults to 0.0
        downsample (bool, optional): Whether to downsample at the end of the layer. Defaults to True
        size (int, optional): Input resolution/sequence length. Defaults to 56
        
    The layer consists of:
    1. A sequence of Swin blocks that alternate between regular and shifted window attention
    2. Optional downsampling via patch merging that reduces spatial dimensions by 2x
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, dropout_rate=0.0,
                 downsample=True, size=56):
        super().__init__()
        self.size = size
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                size=self.size)
            for i in range(depth)
        ])

        if downsample:
            self.downsample = PatchMerging(dim=dim, size=self.size)
        else:
            self.downsample = None

    def forward(self, x, size):
        for block in self.blocks:
            x = block(x, size)
        if self.downsample is not None:
            x, size = self.downsample(x, size)  # Reduce size by 2
        return x, size

#Shape Error
class SwinBlock(nn.Module):
    """Swin Transformer block with window attention and shifted window attention.

    This block contains a window-based multi-head self-attention module followed by an MLP.
    It supports both regular window attention and shifted window attention based on the shift_size parameter.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        window_size (int, optional): Size of attention window. Defaults to 7.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
        dropout_rte (float, optional): Dropout rate. Defaults to 0.0.
        shift_size (int, optional): Size of window shift. Defaults to 0.
        size (int, optional): Input resolution. Defaults to 56.

    Attributes:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        window_size (int): Size of attention window
        shift_size (int): Size of window shift
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        size (int): Input resolution
        norm1 (nn.LayerNorm): Layer normalization before attention
        attn (WindowAttention): Window-based multi-head self-attention module
        norm2 (nn.LayerNorm): Layer normalization before MLP
        mlp (MLP): Multi-layer perceptron module
    """
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0, dropout_rate=0.0,
                 shift_size=0, size=56):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.size = size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, dropout_rate=dropout_rate)

    def forward(self, x, size):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, size, size, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)  # B*num_windows, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)  # B*num_windows, window_size*window_size, C

        x = window_reverse(attn_windows, self.window_size, size)  # B, S, S, C

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, size * size, C)
        x = shortcut + x  # Residual connection
        x = x + self.mlp(self.norm2(x))  # Residual connection with MLP

        return x


class WindowAttention(nn.Module):
    """

    This module implements the window-based multi-head self attention mechanism used in Swin Transformer.
    It divides the input into windows and performs self-attention within each window.

    Args:
        dim (int): Input feature dimension
        window_size (int): Size of each attention window
        num_heads (int): Number of attention heads
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0

    Attributes:
        dim (int): Input feature dimension
        window_size (int): Size of each attention window
        num_heads (int): Number of attention heads
        scale (float): Scaling factor for attention scores
        qkv (nn.Linear): Linear projection for query, key and value
        proj (nn.Linear): Linear projection for output
        proj_drop (nn.Dropout): Dropout layer
    """
    def __init__(self, dim, window_size, num_heads, dropout_rate=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multilayer perceptron used in transformer blocks."""
    def __init__(self, in_features, hidden_features, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging layer to reduce spatial dimensions.

    This layer merges adjacent patches by concatenating their features and projecting to a lower dimension.
    It reduces spatial dimensions by a factor of 2 while increasing the channel dimension.

    Args:
        dim (int): Number of input channels
        size (int): Size of the input feature map (assuming square input)

    Attributes:
        dim (int): Number of input channels
        size (int): Size of the input feature map
        reduction (nn.Linear): Linear layer to reduce concatenated features
        norm (nn.LayerNorm): Layer normalization applied before reduction
    """
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, size):
        B, L, C = x.shape
        assert L == size * size, "Input feature has incorrect size."

        x = x.view(B, size, size, C)
        x0 = x[:, 0::2, 0::2, :]  # B, S/2, S/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, S/2, S/2, 4C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # B, S/2 * S/2, 2C

        size = size // 2  
        return x, size


def window_partition(x, window_size):
    """Partitions feature map into non-overlapping windows."""
    B, S, S_, C = x.shape
    assert S == S_, "Size mismatch in window partition."

    x = x.view(B, S // window_size, window_size, S // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, size):
    """Reverses window partitioning."""
    B = int(windows.shape[0] / (size * size / window_size / window_size))
    x = windows.view(B, size // window_size, size // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, size, size, -1)
    return x
