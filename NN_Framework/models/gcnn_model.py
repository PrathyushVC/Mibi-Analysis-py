import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, InstanceNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader

class GraphConvClassifier(nn.Module):
    """Graph Convolutional Network (GCN) for graph classification.

    This GCN model classifies entire graphs based on node features and graph structure.
    The model includes:
    -  GCN layers to learn node embeddings by aggregating neighboring node features.
    - Global mean pooling to create a graph-level embedding from node embeddings.
    - A fully connected layer for graph classification.

    Attributes:
        conv1 (GCNConv): The first GCN layer.
        conv2 (GCNConv): The second GCN layer.
        relu (ReLU): Activation function applied after each GCN layer.
        fc (Linear): Fully connected layer for classification.
        log_softmax (LogSoftmax): Final activation function to produce class probabilities.
    """
    
    def __init__(self, input_dim, hidden_dims=[128,128], num_classes=2,dropout_rate=0.1,
                 num_gcn_layers=2, pooling="mean", use_batch_norm=True):
        """Initializes the GraphClassifier model with two GCN layers, a fully connected
        layer, and an output layer.

        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden layers.
            num_classes (int): Number of output classes for graph classification.
        """
        super(GraphConvClassifier, self).__init__()

        if pooling not in ["mean", "max"]:
            raise ValueError("Invalid pooling method. Choose 'mean' or 'max'.")
        
        self.num_gcn_layers = num_gcn_layers
        self.pooling = pooling

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dims[-1])

        
        for i in range(len(hidden_dims)):
            in_channels = input_dim if i == 0 else hidden_dims[i - 1]
            out_channels = hidden_dims[i]
            self.convs.append(GCNConv(in_channels, out_channels))
            self.norms.append(InstanceNorm(out_channels))

        self.fc = nn.Linear(hidden_dims[-1], num_classes)
        

        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        """Forward pass through the model to compute class probabilities for input graphs.

        Args:
            x (Tensor): Node features of shape [num_nodes, input_dim].
            edge_index (Tensor): Edge connectivity matrix of shape [2, num_edges].
            batch (Tensor): Batch vector indicating graph membership of each node.

        Returns:
            Tensor: Logarithmic probabilities for each graph's class, with shape [num_graphs, output_dim].
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)

        x = global_mean_pool(x, batch)

        if self.use_batch_norm:
            x = self.batch_norm(x)

        x = self.fc(x)

        return x



if __name__ == "__main__":
    print("Your importing this wrong. Dont do this")