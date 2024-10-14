"""Graph Data Processing Utilities"""

import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from typing import Optional, Any
import numpy.typing as npt


def calculate_normalized_laplacian(adj: npt.NDArray[np.float64]) -> sp.coo_matrix:
    """
    Calculates the normalized Laplacian of an adjacency matrix.

    The normalized Laplacian is given by L = I - D^-1/2 A D^-1/2, where A is the adjacency matrix,
    D is the degree matrix, and I is the identity matrix.

    Parameters
    ----------
    adj : npt.NDArray[np.float64]
        The adjacency matrix.

    Returns
    -------
    sp.coo_matrix
        The normalized Laplacian matrix in sparse COO format.
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))  # Degree matrix (sum of rows)
    d_inv_sqrt = np.power(d + 1e-6, -0.5).flatten()  # D^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # Handle division by zero
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx: npt.NDArray[np.float64]) -> sp.coo_matrix:
    """
    Calculates the random walk matrix of an adjacency matrix.

    The random walk matrix is given by D^-1 A, where A is the adjacency matrix and D is the degree matrix.

    Parameters
    ----------
    adj_mx : npt.NDArray[np.float64]
        The adjacency matrix.

    Returns
    -------
    sp.coo_matrix
        The random walk matrix in sparse COO format.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))  # Degree matrix
    d_inv = np.power(d + 1e-6, -1).flatten()  # D^-1
    d_inv[np.isinf(d_inv)] = 0.0  # Handle division by zero
    d_mat_inv = sp.diags(d_inv)

    # Random walk matrix: D^-1 A
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(
    adj_mx: npt.NDArray[np.float64],
) -> sp.coo_matrix:
    """
    Calculates the reverse random walk matrix of an adjacency matrix.

    The reverse random walk matrix is computed by transposing the adjacency matrix
    and then calculating the random walk matrix.

    Parameters
    ----------
    adj_mx : npt.NDArray[np.float64]
        The adjacency matrix.

    Returns
    -------
    sp.coo_matrix
        The reverse random walk matrix in sparse COO format.
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(
    adj_mx: npt.NDArray[np.float64],
    lambda_max: Optional[float] = 2,
    undirected: bool = True,
) -> sp.csr_matrix:
    """
    Calculates the scaled Laplacian of an adjacency matrix.

    The scaled Laplacian is given by (2 / lambda_max) * L - I, where L is the normalized Laplacian,
    and lambda_max is the largest eigenvalue of L.

    Parameters
    ----------
    adj_mx : npt.NDArray[np.float64]
        The adjacency matrix.
    lambda_max : Optional[float], optional
        The maximum eigenvalue of the Laplacian, by default 2.
    undirected : bool, optional
        Whether the graph is undirected. If True, the adjacency matrix is symmetrized.

    Returns
    -------
    sp.csr_matrix
        The scaled Laplacian in sparse CSR format.
    """
    if undirected:
        adj_mx = np.maximum.reduce(
            [adj_mx, adj_mx.T]
        )  # Symmetrize the adjacency matrix

    # Compute the normalized Laplacian
    L = calculate_normalized_laplacian(adj_mx)

    # Compute the largest eigenvalue if not provided
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)

    # Scaled Laplacian: (2 / lambda_max) * L - I
    L_scaled = (2 / lambda_max * L) - I
    return L_scaled.astype(np.float32)


def load_graph_data(pkl_filename: str) -> tuple[list[str], dict[str, int], np.ndarray]:
    """
    Loads the graph data, including sensor IDs, sensor index mapping, and the adjacency matrix.

    Parameters
    ----------
    pkl_filename : str
        The path to the pickle file containing the graph data.

    Returns
    -------
    tuple[list[str], dict[str, int], np.ndarray]
        sensor_ids: List of sensor IDs.
        sensor_id_to_ind: Dictionary mapping sensor IDs to indices.
        adj_mx: The adjacency matrix.
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj_mx = adj_mx - np.eye(adj_mx.shape[0])  # Remove self-loops
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file: str) -> Any:
    """
    Loads data from a pickle file with support for various encodings.

    Parameters
    ----------
    pickle_file : str
        The path to the pickle file.

    Returns
    -------
    Any
        The loaded pickle data.

    Raises
    ------
    Exception
        If the file cannot be loaded.
    """
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Unable to load data {pickle_file}: {e}")
        raise
    return pickle_data


def calculate_cheb_poly(L: npt.NDArray[np.floa64], Ks: int) -> npt.NDArray[np.floa64]:
    """
    Calculates the Chebyshev polynomials up to order K for the graph Laplacian.

    Parameters
    ----------
    L : npt.NDArray[np.floa64]
        The Laplacian matrix.
    Ks : int
        The order of the Chebyshev polynomials.

    Returns
    -------
    npt.NDArray[np.floa64]
        The Chebyshev polynomials as a list of matrices.
    """
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]  # T_0 = I, T_1 = L

    # Compute T_k = 2L T_(k-1) - T_(k-2) for k >= 2
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])

    return np.asarray(LL)


def sym_adj(adj: npt.NDArray[np.float64]) -> npt.NDArray[np.floa64]:
    """
    Symmetrically normalizes an adjacency matrix.

    The symmetric normalization is given by D^-1/2 A D^-1/2, where A is the adjacency matrix
    and D is the degree matrix.

    Parameters
    ----------
    adj : npt.NDArray[np.float64]
        The adjacency matrix.

    Returns
    -------
    npt.NDArray[np.floa64]
        The symmetrically normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # Degree matrix
    d_inv_sqrt = np.power(rowsum + 1e-6, -0.5).flatten()  # D^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # Handle division by zero
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Symmetric normalization: D^-1/2 A D^-1/2
    return (
        adj.dot(d_mat_inv_sqrt)
        .transpose()
        .dot(d_mat_inv_sqrt)
        .astype(np.float32)
        .todense()
    )


def asym_adj(adj: npt.NDArray[np.float64]) -> npt.NDArray[np.floa64]:
    """
    Asymmetrically normalizes an adjacency matrix.

    The asymmetric normalization is given by D^-1 A, where A is the adjacency matrix and D is the degree matrix.

    Parameters
    ----------
    adj : npt.NDArray[np.float64]
        The adjacency matrix.

    Returns
    -------
    npt.NDArray[np.floa64]
        The asymmetrically normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()  # Degree matrix
    d_inv = np.power(rowsum + 1e-6, -1).flatten()  # D^-1
    d_inv[np.isinf(d_inv)] = 0.0  # Handle division by zero
    d_mat = sp.diags(d_inv)

    # Asymmetric normalization: D^-1 A
    return d_mat.dot(adj).astype(np.float32).todense()
