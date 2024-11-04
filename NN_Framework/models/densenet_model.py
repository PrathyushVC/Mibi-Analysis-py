import torch
import torch.nn as nn

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
        dropout_rate (float, optional): The dropout rate applied after the second convolutional layer. 
                                      Default is 0 (no dropout).

    Forward Pass:
        The forward method applies batch normalization, ReLU activation, and two convolutional layers 
        sequentially. If a dropout rate is specified, dropout is applied before concatenating the 
        output with the input.

    Returns:
        Tensor: The output tensor after applying the operations defined in the layer.
    """

    def __init__(self, in_channels, growth_rate, bn_size=4, dropout_rate=0):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        if dropout_rate > 0:
           self.dropout=nn.Dropout(p=dropout_rate)
        else:
            self.dropout=None


    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.dropout is not None:
            out = self.dropout(out)
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
        dropout_rate (float, optional): The dropout rate for the layers. Default is 0.
    """

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, dropout_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,  \
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate)
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

        self.transition=nn.Sequential( 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    """
    Args:
        num_init_features (int): Number of filters in the initial convolution layer.
        growth_rate (int): How many filters to add each DenseLayer.
        block_config (tuple): Number of layers in each DenseBlock.
        num_classes (int): Number of classification classes.
        bn_size (int): Multiplicative factor for bottleneck layers.
        dropout_rate (float): Dropout rate after each DenseLayer.
        input_channels (int): Number of input image channels.
    """
    def __init__(self, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, bn_size=4, dropout_rate=0, input_channels=3):
        super(DenseNet, self).__init__()

        self.hparams = {
            'num_init_features': num_init_features,
            'growth_rate': growth_rate,
            'block_config': block_config,
            'num_classes': num_classes,
            'bn_size': bn_size,
            'dropout_rate': dropout_rate,
            'input_channels': input_channels}

        self.features = nn.Sequential()

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
                               dropout_rate=dropout_rate)
            
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(in_channels=num_features,
                                        out_channels=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.fc = nn.Linear(num_features, num_classes) 
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
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


