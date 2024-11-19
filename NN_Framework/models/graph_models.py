import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, InstanceNorm, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GraphConvClassifier(nn.Module):
    """Graph Convolutional Network (GCN) for graph classification.

    This GCN model classifies entire graphs based on node features and graph structure.
    The model includes:
    - Two GCN layers to learn node embeddings by aggregating neighboring node features.
    - Global mean pooling to create a graph-level embedding from node embeddings.
    - A fully connected layer for graph classification.

    Attributes:
        conv1 (GCNConv): The first GCN layer.
        conv2 (GCNConv): The second GCN layer.
        relu (ReLU): Activation function applied after each GCN layer.
        fc (Linear): Fully connected layer for classification.
        log_softmax (LogSoftmax): Final activation function to produce class probabilities.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=2,dropout_rate=0.1):
        """Initializes the GraphClassifier model with two GCN layers, a fully connected
        layer, and an output layer.

        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden layers.
            num_classes (int): Number of output classes for graph classification.
        """
        super(GraphConvClassifier, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = InstanceNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = InstanceNorm(hidden_dim)


        self.fc = nn.Linear(hidden_dim, num_classes)

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
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    print("Your importing this wrong. Dont do this")