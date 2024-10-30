import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, DataLoader
from scipy.spatial import distance_matrix



class CellTypeEmbedding(nn.Module):#Move to model module?
    """
    CellTypeEmbedding is a PyTorch neural network module that creates embeddings for cell types.

    Args:
        num_classes (int): The number of unique cell types.
        embedding_dim (int): The dimension of the embedding space.

    Methods:
        forward(x):
            Computes the forward pass of the embedding layer, returning the embeddings for the input indices.
    """
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

def create_global_cell_mapping(df, cell_type_col):
    """
    Creates a global mapping of cell types to unique indices.

    Args:
        df (DataFrame): The input DataFrame containing cell type information.
        cell_type_col (str): The name of the column in the DataFrame that contains cell type information.

    Returns:
        dict: A dictionary mapping each unique cell type to its corresponding index.
    """

    unique_cell_types = df[cell_type_col].unique().to_list() 
    cell_type_to_index = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
    return cell_type_to_index

def map_cell_types_to_indices(df, cell_type_col, cell_type_to_index):
    """Map cell types to their corresponding indices."""
    df=df.with_columns(pl.col(cell_type_col).replace(cell_type_to_index).cast(pl.Int32).alias(cell_type_col+'_map'))
    return df
def create_hetero_graph(df, expressions, cell_type_col=None, radius=50, 
                        x_pos='centroid-0', y_pos='centroid-1', 
                        fov_col='fov', group_col='Group', binarize=False, embedding_dim=4):
    data = HeteroData()

    if cell_type_col:
        num_cell_types = df[cell_type_col].n_unique()
        cell_type_embedding = CellTypeEmbedding(num_cell_types, embedding_dim)
        cell_type_index = create_global_cell_mapping(df, cell_type_col)
    else:
        cell_type_embedding = None


    #Determine how to do this in polars directly

    for group_key,df_fov in df.group_by(fov_col):
        

        try:
            print(df_fov.select(expressions).to_numpy())
            data_to_numpy=df_fov.select(expressions).to_numpy()
        except Exception as e:
            raise TypeError(f"Columns contained non numerical types can not be converted to numpy:  {str(e)}")
        
        node_features = torch.tensor(data_to_numpy, dtype=torch.float32)
        centroids = df_fov.select([x_pos, y_pos]).to_numpy()

        if cell_type_col:
            # Map cell types to indices and generate embeddings
            cell_type_indices = torch.tensor(
                map_cell_types_to_indices(df_fov, cell_type_col, cell_type_index), 
                dtype=torch.long
            )
            cell_embeddings = cell_type_embedding(cell_type_indices)

            
            node_features = torch.cat([node_features, cell_embeddings], dim=1)

        
        dist_matrix = distance_matrix(centroids, centroids)
        edge_index = np.array((dist_matrix < radius).nonzero())
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops

        
        data['cell'].x = node_features
        data['cell', 'connected_to', 'cell'].edge_index = edge_index
        edge_attr = torch.tensor(dist_matrix[edge_index[0], edge_index[1]], dtype=torch.float).unsqueeze(-1)
        data['cell', 'connected_to', 'cell'].edge_attr = edge_attr

        
        group_value = df_fov[group_col][0]
        graph_label = _binarize_group(group_value) if binarize else _quad_group(group_value)
        data['graph_label'] = torch.tensor([graph_label], dtype=torch.long)

    return data

def _binarize_group(group_data):
    """Binarize group labels."""
    if str(group_data) in ['G1', 'G4']:
        return 1
    elif str(group_data) in ['G2', 'G3']:
        return 0
    else:
        raise ValueError("Unexpected group data")

def _quad_group(group_data):
    """Map groups to 4-class labels."""
    group_mapping = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3}
    if str(group_data) in group_mapping:
        return group_mapping[str(group_data)]
    else:
        raise ValueError("Unexpected group data")

def dataset_overlap(df1, df2, col):
    """Check for overlapping patient numbers."""
    set1, set2 = set(df1[col].to_list()), set(df2[col].to_list())
    if set1 & set2:
        print("There is an overlap in patient numbers.")
    else:
        print("No overlap in patient numbers.")

if __name__ == "__main__":
    # Load data
    df = pl.read_csv(r"D:\MIBI-TOFF\Data_For_Amos\cleaned_expression_with_both_classification_prob_spatial_30_08_24.csv")
    expressions = ['CD45']

    df = df[~df['pred'].isin(['Unidentified', 'Immune'])]#Remove confounding cells

    # Create graphs
    graphs = create_hetero_graph(df, expressions, cell_type_col='pred', radius=50)
    torch.save(graphs, 'fov_graphs.pt')
    print(f"Saved {len(graphs)} graphs.")

    # Load and test DataLoader
    loaded_graphs = torch.load('fov_graphs.pt')
    loader = DataLoader(loaded_graphs, batch_size=2, shuffle=True)

    for batch in loader:
        print(batch)
