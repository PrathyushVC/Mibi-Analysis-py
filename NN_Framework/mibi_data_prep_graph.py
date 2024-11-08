import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import distance_matrix

#TODO edge values hard set to a single value and the model load through data loader



class CellTypeEmbedding(nn.Module):  # Move to model module?
    """A PyTorch module that creates embeddings for cell types.

    This module converts categorical cell type labels into learned vector embeddings
    that can be used as node features in a graph neural network.

    Args:
        num_classes (int): Number of unique cell types to embed
        embedding_dim (int): Dimension of the embedding vectors

    Returns:
        torch.Tensor: Embedded vectors of shape (batch_size, embedding_dim)
    """
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


def create_global_cell_mapping(df, cell_type_col):
    unique_cell_types = df[cell_type_col].unique().to_list()
    cell_type_to_index = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
    return cell_type_to_index


def map_cell_types_to_indices(df, cell_type_col, cell_type_to_index):
    df = df.with_columns(pl.col(cell_type_col).replace(cell_type_to_index).cast(pl.Int32).alias(cell_type_col + '_int_map'))
    return df


def create_graph(df, expressions, cell_type_col=None, radius=50, x_pos='centroid-0', y_pos='centroid-1', fov_col='fov', group_col='Group', binarize=False, embedding_dim=4):
    """Creates heterogeneous graphs from cell data.

    Constructs graph representations of cell data where cells are nodes connected by edges
    based on spatial proximity. Optionally includes cell type embeddings as node features.

    Args:
        df (polars.DataFrame): Input dataframe containing cell data
        expressions (list): Column names of expression values to use as node features
        cell_type_col (str, optional): Column name containing cell type labels. If provided,
            cell type embeddings will be created. Defaults to None.
        radius (float, optional): Maximum distance between cells to create an edge.
            Defaults to 50.
        x_pos (str, optional): Column name for x coordinates. Defaults to 'centroid-0'.
        y_pos (str, optional): Column name for y coordinates. Defaults to 'centroid-1'.
        fov_col (str, optional): Column name for field of view IDs. Defaults to 'fov'.
        group_col (str, optional): Column name for group labels. Defaults to 'Group'.
        binarize (bool, optional): Whether to binarize group labels. Defaults to False.
        embedding_dim (int, optional): Dimension of cell type embeddings. Defaults to 4.

    Returns:
        list[torch_geometric.data.Data]: List of heterogeneous graphs, one per FOV.
            Each graph contains:
            - Node features (x): Expression values and optional cell type embeddings
            - Edge indices: Pairs of connected cell indices
            - Edge attributes: Distances between connected cells
            - Graph label: Group classification label
    """
    graphs = []

    # Initialize embedding only if cell_type_col is specified
    if cell_type_col:
        num_cell_types = df[cell_type_col].n_unique()
        cell_type_embedding = CellTypeEmbedding(num_cell_types, embedding_dim)
        cell_type_index = create_global_cell_mapping(df, cell_type_col)
    else:
        cell_type_embedding = None

    for group_key, df_fov in df.group_by(fov_col):
        data = Data()
        data_to_numpy = df_fov.select(expressions).to_numpy()
        node_features = torch.tensor(data_to_numpy, dtype=torch.float32)
        centroids = df_fov.select([x_pos, y_pos]).to_numpy()

        if cell_type_col:
            df_fov = map_cell_types_to_indices(df_fov, cell_type_col, cell_type_index)
            cell_type_indices = torch.tensor(df_fov[cell_type_col + '_int_map'].to_numpy(), dtype=torch.long)
            cell_embeddings = cell_type_embedding(cell_type_indices)
            node_features = torch.cat([node_features, cell_embeddings], dim=1)

        dist_matrix = distance_matrix(centroids, centroids)
        edge_index = np.array((dist_matrix < radius).nonzero())
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove loops

        data.x = node_features
        data.edge_index = edge_index
        edge_attr = torch.tensor(dist_matrix[edge_index[0], edge_index[1]], dtype=torch.float).unsqueeze(-1)
        data.edge_attr = edge_attr

        group_value = df_fov[group_col][0]
        graph_label = _binarize_group(group_value) if binarize else _quad_group(group_value)
        data.y = torch.tensor([graph_label], dtype=torch.long)

        print(f"Node features shape: {node_features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Distance matrix shape: {dist_matrix.shape}")
        print(f"Edge attributes shape: {edge_attr.shape}")


        graphs.append(data)
        del edge_attr, edge_index, node_features, centroids, dist_matrix
    return graphs

def _binarize_group(group_data):
    
    if str(group_data) in ['G1', 'G4']:
        return 1
    elif str(group_data) in ['G2', 'G3']:
        return 0
    else:
        raise ValueError("Unexpected group data")

def _quad_group(group_data):
    
    group_mapping = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3}
    if str(group_data) in group_mapping:
        return group_mapping[str(group_data)]
    else:
        raise ValueError("Unexpected group data")

def dataset_overlap(df1, df2, col):
   
    set1, set2 = set(df1[col].to_list()), set(df2[col].to_list())
    if set1 & set2:
        print("There is an overlap in patient numbers.")
    else:
        print("No overlap in patient numbers.")

def print_data_details(data):
    """Prints detailed information about a PyG Data object.

    Args:
        data (torch_geometric.data.Data): A PyG Data object containing graph information.
            Expected to have node features (x), edge indices (edge_index), and optionally
            edge attributes (edge_attr) and graph labels (graph_label).

    Prints:
        - Number of nodes and first 5 rows of node features
        - Number of edges and first 5 edge index pairs
        - First 5 edge attributes (if present)
        - Graph label (if present)
    """
    print("Graph Summary:")
    print("\nNode Features:")
    num_nodes = data.x.size(0)
    features = data.x
    print(f"  - Num nodes: {num_nodes}")
    print(f"    Features (first 5 rows):\n{features[:5]}")

    print("\nEdge Index:")
    num_edges = data.edge_index.size(1)
    print(f"  - Num edges: {num_edges}")
    print(f"    Edge Index (first 5 pairs):\n{data.edge_index[:, :5]}")

    if 'edge_attr' in data:
        edge_attr = data.edge_attr
        print(f"    Edge Attributes (first 5 rows):\n{edge_attr[:5]}")

    if 'graph_label' in data:
        graph_label = data.graph_label
        print(f"\nGraph Label:\n{graph_label}")


def remapping(df, column_name):
    # Define the mapping as a dictionary
    group_mapping = {
        'CD4 T cell': 'CD4_T_cell',
        'Memory_CD4_T_Cells': 'Memory_CD4_T_Cells',
        'CD8 T cell': 'CD8_T_cell',
        'CD4 APC': 'DCs',
        'CD4 Treg': 'CD4_Treg',
        'CD3 only': 'CD3',
        'B cell': 'B_Cells',
        'Follicular_Germinal_B_Cell': 'Germinal_Center_B_Cell',
        'CD20_neg_B_cells': 'B_Cells',
        'DC sign Mac': 'MAC',
        'CD206_Mac': 'MAC',
        'CD68_Mac': 'MAC',
        'Mac': 'MAC',
        'DCs': 'DCs',
        'CD14_CD11c_DCs': 'DCs',
        'CD11_CD11c_DCsign_DCs': 'DCs',
        'Mono_CD14_DR': 'MAC',
        'tumor': 'tumor',
        'Hevs': 'Hevs',
        'Collagen_sma': 'Stroma',
        'SMA': 'Stroma',
        'Collagen': 'Stroma',
        'Neutrophil': 'Neutrophil',
        'Tfh': 'CD4_T_cell',
        'NK cell': 'NK cell',
        'Immune': 'Immune',
        'Unidentified': 'Unidentified',
        'blood vessels': 'blood vessels',
    }
    
    if isinstance(df, pd.DataFrame):  
        df['remapped'] = df[column_name].map(group_mapping).fillna('Unidentified')
    elif isinstance(df, pl.DataFrame): 
        df = df.with_columns(pl.col(column_name).replace(group_mapping).alias('remapped'))
    else:
        raise ValueError("Input DataFrame must be either a Pandas or Polars DataFrame.")

    return df



if __name__ == "__main__":
    import torch
    import torch_geometric
    print(torch.__version__)
    print(torch_geometric.__version__)
    
    print("Running mibi_data_prep_graph dirrectly. Are you sure this is a good idea?")
    df = pl.read_csv(r"D:\MIBI-TOFF\Data_For_Amos\cleaned_expression_with_both_classification_prob_spatial_30_08_24.csv")
    expressions = ['CD45']

    df = df.filter(~pl.col('pred').is_in(['Unidentified', 'Immune']))#Remove confounding cells

    df=remapping(df=df, column_name='pred')#remap larger cell name list to smaller one 

    
    graphs = create_graph(df, expressions, cell_type_col='remapped', radius=50)
    torch.save(graphs, r"D:\MIBI-TOFF\Scratch\fov_graphs.pt")
    print(f"Saved {len(graphs)} graphs.")

    # Load and test DataLoader
    loaded_graphs = torch.load(r"D:\MIBI-TOFF\Scratch\fov_graphs.pt")
    print(f"loaded {len(loaded_graphs)} graphs.")
    for graph in loaded_graphs:
        print_data_details(graph)
        
    loader = DataLoader(loaded_graphs, batch_size=1, shuffle=True)
    print(loader)
#
    for batch in loader:
        print(batch)
