#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from ot import emd

def cell_mappings(sn_data, st_data, rmge, device):
    """
    Computes cell mappings between sn_data and st_data using optimal transport.

    Args:
        sn_data: The single-nucleus dataset (sn_data).
        st_data: The spatial transcriptomics dataset (st_data).
        rmge: The trained RMGE model.
        device: The device (CPU or GPU) where the model should run.

    Returns:
        cell_mappings (list): A list of tuples with cell mappings (sn_cell_id, st_cell_id).
        total_transport (float): The total transport cost.
    """
    cell_mappings = []
    rmge.model.eval()

    # Convert sn_data and st_data into tensors
    sn_cell_tensor = torch.tensor(sn_data.X.astype(np.float32)).to(device)
    st_cell_tensor = torch.tensor(st_data.X.astype(np.float32)).to(device)

    # Compute projections for both sn_data and st_data
    with torch.no_grad():
        sn_projection, _ = rmge.model(sn_cell_tensor)
        st_projection, _ = rmge.model(st_cell_tensor)

    # Create DataFrames for the projections
    sn_score = pd.DataFrame(sn_projection.cpu().numpy(), columns=rmge.class_names, index=sn_data.obs.index)
    st_score = pd.DataFrame(st_projection.cpu().numpy(), columns=rmge.class_names, index=st_data.obs.index)

    # Rank cells by their projections
    sn_rank_indices = np.argsort(sn_score.values, axis=1)[:, ::-1]
    st_rank_indices = np.argsort(st_score.values, axis=1)[:, ::-1]

    # Convert rank indices to DataFrame
    sn_rank_df = pd.DataFrame(sn_rank_indices, index=sn_score.index)
    st_rank_df = pd.DataFrame(st_rank_indices, index=st_score.index)

    # Get main cell types (the first column of the rank DataFrame)
    sn_main_types = sn_rank_df.iloc[:, 0].values
    st_main_types = st_rank_df.iloc[:, 0].values

    # Extract the features (remaining columns) for the optimal transport computation
    sn_features = sn_rank_df.iloc[:, 1:].values
    st_features = st_rank_df.iloc[:, 1:].values

    # Get unique main cell types
    unique_main_types = np.unique(sn_main_types)

    # Initialize total transport cost
    total_transport = 0

    # Loop over each main type to calculate optimal transport
    for main_type in unique_main_types:
        # Find indices of cells belonging to the current main type in both datasets
        sn_main_indices = np.where(sn_main_types == main_type)[0]
        st_main_indices = np.where(st_main_types == main_type)[0]

        if st_main_indices.size == 0:
            print(f"{main_type} is empty")
            continue

        # Subset the features for the current main type
        sn_subset = sn_features[sn_main_indices, :]
        st_subset = st_features[st_main_indices, :]

        # Compute the difference matrix
        num_cells_sn = sn_subset.shape[0]
        num_cells_st = st_subset.shape[0]
        num_cols = sn_subset.shape[1]
        
        diff_matrix = sn_subset[:, None, :] - st_subset[None, :, :]
        
        # Create a mask to find non-zero differences
        non_zero_mask = diff_matrix != 0
        first_non_zero_indices = np.where(np.any(non_zero_mask, axis=2), np.argmax(non_zero_mask, axis=2), diff_matrix.shape[2])

        # Calculate the distance matrix
        distance_matrix = num_cols - first_non_zero_indices.astype(int)

        # Assume equal weights for each cell
        weights_sn = np.array([1 / num_cells_sn] * num_cells_sn)
        weights_st = np.array([1 / num_cells_st] * num_cells_st)
        weights_sn /= weights_sn.sum()
        weights_st /= weights_st.sum()

        # Compute the transport matrix using optimal transport
        transport_matrix = emd(weights_sn, weights_st, distance_matrix)
        total_transport += np.sum(transport_matrix * distance_matrix)

        # Loop over the transport matrix to generate the cell mappings
        for i in range(transport_matrix.shape[1]):
            best_match_sn_index = np.argmax(transport_matrix[:, i])

            # Get the corresponding sn and st cell indices
            sn_cell_index = sn_main_indices[best_match_sn_index]
            st_cell_index = st_main_indices[i]

            # Retrieve the actual cell IDs from the rank DataFrame
            sn_cell_id = sn_rank_df.index[sn_cell_index]
            st_cell_id = st_rank_df.index[st_cell_index]

            # Append the mapping to the result list
            cell_mappings.append((sn_cell_id, st_cell_id))

    # Return the computed cell mappings and total transport cost
    return cell_mappings, total_transport