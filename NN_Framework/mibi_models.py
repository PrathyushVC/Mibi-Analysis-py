import torch
import torch.nn as nn
import torch.nn.functional as F
#TODO test SWINTRANSFORMER

class ViTClassifier(nn.Module):
    def __init__(self, img_size_x, img_size_y, in_channels=1, num_classes=4, 
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


class DenseLayer(nn.Module):
    """
    DenseLayer is a building block for DenseNet architectures. It consists of two convolutional layers 
    with batch normalization and ReLU activation in between. The output of the second convolutional layer 
    is concatenated with the input, allowing for feature reuse and promoting gradient flow.

    Args:
        in_channels (int): The number of input channels to the layer.
        growth_rate (int): The number of output channels for the second convolutional layer.
        bn_size (int, optional): The size of the bottleneck layer, which determines the number of output 
                                  channels from the first convolutional layer. Default is 4.
        drop_rate (float, optional): The dropout rate applied after the second convolutional layer. 
                                      Default is 0 (no dropout).

    Forward Pass:
        The forward method applies batch normalization, ReLU activation, and two convolutional layers 
        sequentially. If a dropout rate is specified, dropout is applied before concatenating the 
        output with the input.

    Returns:
        Tensor: The output tensor after applying the operations defined in the layer.
    """

    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = torch.cat([x, out], 1)
        return out


#Copied and cleaned from older INVent project folders
class DenseBlock(nn.Module):
    """
    A DenseBlock consists of multiple DenseLayers.

    Args:
        num_layers (int): The number of layers in the DenseBlock.
        in_channels (int): The number of input channels to the first layer.
        growth_rate (int): The number of output channels for each DenseLayer.
        bn_size (int, optional): The size of the bottleneck layer. Default is 4.
        drop_rate (float, optional): The dropout rate for the layers. Default is 0.
    """

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,  # Incremental input channels
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    """
    TransitionLayer is a layer that reduces the number of feature maps and downsamples the input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the transition.
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, bn_size=4, drop_rate=0, input_channels=3):
        """
        Args:
            num_init_features (int): Number of filters in the initial convolution layer.
            growth_rate (int): How many filters to add each DenseLayer.
            block_config (tuple): Number of layers in each DenseBlock.
            num_classes (int): Number of classification classes.
            bn_size (int): Multiplicative factor for bottleneck layers.
            drop_rate (float): Dropout rate after each DenseLayer.
            input_channels (int): Number of input image channels.
        """
        super(DenseNet, self).__init__()

        self.hparams = {
            'num_init_features': num_init_features,
            'growth_rate': growth_rate,
            'block_config': block_config,
            'num_classes': num_classes,
            'bn_size': bn_size,
            'drop_rate': drop_rate,
            'input_channels': input_channels
        }

        self.features = nn.Sequential()
        # Initial convolution layer
        self.features.add_module('conv0', nn.Conv2d(input_channels, num_init_features,
                                                   kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               in_channels=num_features,
                               growth_rate=growth_rate,
                               bn_size=bn_size,
                               drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(in_channels=num_features,
                                        out_channels=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes) 
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

#Double check
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                nn.init.constant_(m.bias.data, 0)


