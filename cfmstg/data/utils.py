"""Dataset Utils"""
import numpy as np
import numpy.typing as npt


def search_recent_data(
    X: npt.NDArray[np.float64], label_start_idx: int, T_p: int, T_h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """This function searches for recent historical and future data indices
    in the dataset for time-series prediction tasks. It is typically used
    to gather data for training models that make predictions based on a
    sequence of historical data.

    Parameters
    ----------
    X : npt.NDArray[np.float64]
        Node features of shape [T, V, D], where
        - T is the number of time steps
        - V is the number of nodes
        - D is the node feature dimensions
    label_start_idx : int
        The starting index in the dataset where the target
        label (the prediction to be made) starts.
    T_p : int
        The number of time steps to predict (the prediction horizon).
    T_h : int
        The number of historical time steps to consider (the look-back period).

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]] | None
        A tuple of two pairs of indices: ((start_idx, end_idx), (label_start_idx, label_end_idx))
        - (start_idx, end_idx): Indices for the historical data (input).
        - (label_start_idx, label_end_idx): Indices for the future/prediction data (target).

        Returns None if the indices are invalid (out of bounds).
    """
    # Check if prediction range exceeds the available data
    if label_start_idx + T_p > len(X):
        return None

    # Calculate the start and end indices for historical and prediction ranges
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p

    # Check if indices are valid (non-negative)
    if start_idx < 0 or end_idx < 0:
        return None

    # Return the tuple of index ranges for historical data and prediction data
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


def search_multihop_neighbor(
    A: npt.NDArray[np.float64], hops: int = 5
) -> npt.NDArray[np.float64]:
    """This function computes the multi-hop neighborhood matrix for nodes in a graph,
    where each node is connected to its neighbors up to a specified number of hops.
    It creates a matrix where the value at each position (i, j) represents the minimum
    number of hops between node i and node j, up to the given number of hops.


    Parameters
    ----------
    A : npt.NDArray[np.float64]
        The adjacency matrix of the graph, where adj[i, j] = 1 indicates
        a direct connection between nodes i and j, and 0 otherwise.
        Shape: (V, V)
    hops : int, optional
        The maximum number of hops to consider for determining node connectivity,
        by default 5

    Returns
    -------
    npt.NDArray[np.float64]
        A 3D matrix where each element hop_arr[i, j, 0] indicates the number of hops
        between node i and node j. If a node is not reachable within the given number
        of hops, the value is set to (hops + 1).
    """
    # Number of nodes in the graph
    node_cnt = A.shape[0]
    # Initialize hop array
    hop_arr = np.zeros((A.shape[0], A.shape[0]))

    # Loop over all nodes
    for h_idx in range(node_cnt):
        # Initialize neighbors at 0 steps (self)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]
        # Initialize all hops as -1 (unreachable)
        hop_arr[h_idx, :] = -1
        # Set the hop from node to itself to 0
        hop_arr[h_idx, h_idx] = 0

        # Iterate over the hop count (1-hop, 2-hops, ..., max hops)
        for hop_idx in range(hops):
            # To store neighbors of the current step
            tmp_step_node = []
            # To store new nodes found in the kth step
            tmp_step_node_kth = []

            for tmp_nei_node in tmp_neibor_step:
                # Find the direct neighbors of the current node (one-step neighbors)
                tmp_neibor_step = list(np.argwhere(A[tmp_nei_node] == 1).flatten())
                tmp_step_node += tmp_neibor_step

                # Exclude nodes that have already been visited in previous steps
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)

                # Update the list of nodes that have been visited so far
                tmp_h_node += tmp_neibor_step

            # Move to the next set of neighbors for the next hop
            tmp_neibor_step = tmp_step_node_kth.copy()

            # For all nodes found in the kth step, set the hop count
            all_spatial_node = list(set(tmp_neibor_step))
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1

    # Expand the hop array to be 3-dimensional for compatibility with further processing
    return hop_arr[:, :, np.newaxis]
