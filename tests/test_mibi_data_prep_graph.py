import pytest
import torch
import polars as pl
from torch_geometric.data import Data
from NN_Framework.mibi_data_prep_graph import (
    create_global_cell_mapping,
    map_cell_types_to_indices,
    create_graph,
    _binarize_group,
    _quad_group,
    remapping,
    print_data_details
)
# Sample data for testing
@pytest.fixture
def sample_data():
    return pl.DataFrame({
        'centroid-0': [0, 1, 2],
        'centroid-1': [0, 1, 2],
        'pred': ['CD4 T cell', 'CD8 T cell', 'B cell']
    })

def test_create_global_cell_mapping(sample_data):
    mapping = create_global_cell_mapping(sample_data, 'pred')
    assert mapping == {'CD4 T cell': 0, 'CD8 T cell': 1, 'B cell': 2}

def test_map_cell_types_to_indices(sample_data):
    cell_type_to_index = create_global_cell_mapping(sample_data, 'pred')
    mapped_df = map_cell_types_to_indices(sample_data, 'pred', cell_type_to_index)
    assert 'pred_int_map' in mapped_df.columns

def test_create_graph_empty_df():
    empty_df = pl.DataFrame(columns=['centroid-0', 'centroid-1', 'pred'])
    graphs = create_graph(empty_df, ['centroid-0', 'centroid-1'], cell_type_col='pred')
    assert graphs == []

def test_create_graph_with_invalid_radius(sample_data):
    with pytest.raises(ValueError):
        create_graph(sample_data, ['centroid-0', 'centroid-1'], radius=-1)

def test_binarize_group():
    assert _binarize_group('G1') == 1
    assert _binarize_group('G2') == 0
    with pytest.raises(ValueError):
        _binarize_group('G5')

def test_quad_group():
    assert _quad_group('G1') == 0
    assert _quad_group('G4') == 3
    with pytest.raises(ValueError):
        _quad_group('G5')

def test_remapping(sample_data):
    remapped_df = remapping(sample_data, 'pred')
    assert 'remapped' in remapped_df.columns
    assert remapped_df['remapped'][0] == 'CD4_T_cell'

def test_print_data_details():
    data = Data(x=torch.tensor([[1.0], [2.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    print_data_details(data)  # This will print the details, no assertion needed

# To run the tests, use the command: pytest <filename>.py