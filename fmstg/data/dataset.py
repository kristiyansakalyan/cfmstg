"""Torch Dataset Implementation"""

from typing import Literal
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from fmstg.data.utils import search_multihop_neighbor, search_recent_data
from torch.utils.data import Dataset

KnownDatasets = Literal["PEMS", "AIR", "Metro"]

@dataclass
class CleanDatasetConfig:
    # Data related
    data_name: KnownDatasets | str
    node_features_filepath: str
    adjacency_matrix_filepath: str

    # Training related
    val_start_idx: int

    # ST-GCN related!=
    alpha: float | None = None
    t_size: int | None = None


class CleanDataset:
    """
    A class used to clean and preprocess the dataset for graph-based time-series prediction tasks.

    Attributes
    ----------
    data_name : KnownDatasets | str
        Name of the dataset.
    adjacency_matrix_filepath : str
        Path to the file containing adjacancy matrix.
    node_features_filepath : str
        Path to the file containing feature data.
    val_start_idx : int
        Index where the validation dataset starts.
    adj : npt.NDArray[np.float64]
        Adjacency matrix representing spatial relationships between nodes.
    label : npt.NDArray[np.float64]
        Normalized labels for the dataset.
    feature : npt.NDArray[np.float64]
        Normalized features for the dataset.
    spatial_distance : npt.NDArray[np.int32]
        Multi-hop spatial distance matrix between nodes (for spatial GCN models).
    range_mask : npt.NDArray[np.int32]
        Mask for interaction ranges based on multi-hop spatial distance.

    TODO: What are these 2 parameters used for?
    alpha: float
    t_size: int

    mean: float
        Mean of the node features.
    std: float
        Standard deviation of the node features.
    """

    def __init__(self: "CleanDataset", config: CleanDatasetConfig):
        """
        Initializes the dataset, reads the data, and calculates spatial distances for graph models.

        Parameters
        ----------
        config : CleanDatasetConfig
            Configuration object containing paths, parameters, and model settings.
        """
        self.data_name = config.data_name
        self.adjacency_matrix_filepath = config.adjacency_matrix_filepath
        self.node_features_filepath = config.node_features_filepath
        self.val_start_idx = config.val_start_idx

        # Load A and X
        self.adj = np.load(self.adjacency_matrix_filepath)
        self.label, self.feature = self.read_data()

        # For ST-GCN (Spatio-Temporal Graph Convolutional Networks)
        if config.model.get("alpha", None) is not None:
            self.alpha = config.alpha
            self.t_size = config.t_size
            self.spatial_distance = search_multihop_neighbor(self.adj, hops=self.alpha)
            self.range_mask = self.interaction_range_mask(
                hops=self.alpha, t_size=self.t_size
            )

    def read_data(self):
        """
        Reads and preprocesses feature data based on dataset type.

        Returns
        -------
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
            Normalized labels and features.
        """
        if "PEMS" in self.data_name:
            data = np.expand_dims(np.load(self.node_features_filepath)[:, :, 0], -1)
        elif "AIR" in self.data_name:
            data = np.expand_dims(np.load(self.node_features_filepath)[:, :, 0], -1)
            data = np.nan_to_num(data, nan=0)
        elif "Metro" in self.data_name:
            data = np.expand_dims(np.load(self.node_features_filepath)[:, :, 0], -1)
            data = np.nan_to_num(data, nan=0)
        else:
            data = np.load(self.node_features_filepath)

        return self.normalization(data).astype("float32"), self.normalization(
            data
        ).astype("float32")

    def normalization(
        self, feature: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Normalizes the feature data based on training set mean and standard deviation.

        Parameters
        ----------
        feature : npt.NDArray[np.float64]
            Input feature array to be normalized.

        Returns
        -------
        npt.NDArray[np.float64]
            Normalized feature array.
        """
        train = feature[: self.val_start_idx]
        mean = np.mean(train)
        std = np.std(train)
        self.mean = mean
        self.std = std
        return (feature - mean) / std

    def reverse_normalization(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Reverts the normalization process to return the original data values.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Normalized data to revert.

        Returns
        -------
        npt.NDArray[np.float64]
            Original data before normalization.
        """
        return self.mean + self.std * x

    def interaction_range_mask(
        self, hops: int = 2, t_size: int = 3
    ) -> npt.NDArray[np.int32]:
        """
        Creates a mask for interaction ranges based on spatial distances in multiple hops.

        Parameters
        ----------
        hops : int, optional
            The maximum number of hops to consider for the spatial mask, by default 2.
        t_size : int, optional
            The number of time steps, by default 3.

        Returns
        -------
        npt.NDArray[np.int32]
            Interaction range mask for graph convolution operations.
        """
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()] * t_size, axis=-1)  # V, tV


@dataclass
class TrafficDatasetConfig:
    T_h: int
    T_p: int
    V: int
    points_per_hour: int


class TrafficDataset(Dataset):
    """
    A PyTorch Dataset class for loading and accessing traffic time-series data.

    Attributes
    ----------
    T_h : int
        Number of historical time steps to use as input.
    T_p : int
        Number of future time steps to predict.
    V : int
        Number of nodes in the dataset.
    points_per_hour : int
        Number of data points collected per hour.
    data_range : tuple[int, int]
        The range of time indices to consider for the dataset.
    label : npt.NDArray[np.float64]
        Label data for the dataset.
    feature : npt.NDArray[np.float64]
        Feature data for the dataset.
    idx_lst : list
        List of valid indices for extracting data samples.
    """

    def __init__(
        self,
        clean_data: CleanDataset,
        data_range: tuple[int, int],
        config: TrafficDatasetConfig,
    ) -> None:
        """
        Initializes the TrafficDataset by extracting label and feature data from CleanDataset.

        Parameters
        ----------
        clean_data : CleanDataset
            Cleaned and preprocessed dataset object.
        data_range : tuple[int, int]
            The range of time indices to consider for data sampling.
        config : TrafficDatasetConfig
            Configuration object containing parameters for the dataset.
        """
        self.T_h = config.T_h
        self.T_p = config.T_p
        self.V = config.V
        self.points_per_hour = config.points_per_hour
        self.data_range = data_range
        self.data_name = clean_data.data_name

        # (T_total, V, D)
        self.label = np.array(clean_data.label)
        # (T_total, V, D)
        self.feature = np.array(clean_data.feature)

        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print("Sample num:", len(self.idx_lst))

    def __getitem__(self, index: int) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
    ]:
        """
        Returns the label, node features, and time positions for a given sample index.

        Parameters
        ----------
        index : int
            The sample index to retrieve.

        Returns
        -------
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int32]]
            Tuple containing:
            - label: Predicted values.
            - node_feature: Features corresponding to the node data.
            - pos_w: Day of the week position.
            - pos_d: Time of day position.
        """
        recent_idx = self.idx_lst[index]

        # Extract label and feature data
        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]

        # Get temporal positions (day of week and time of day)
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)

        return label, node_feature, pos_w, pos_d

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.idx_lst)

    def get_time_pos(
        self, idx: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Calculates the day of the week and time of day for the sample.

        Parameters
        ----------
        idx : int
            The starting time index.

        Returns
        -------
        tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]
            - pos_w: Day of the week.
            - pos_d: Time of day.
        """
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # Day of the week
        pos_d = idx % (self.points_per_hour * 24)  # Time of day
        return pos_w, pos_d

    def get_idx_lst(self) -> list:
        """
        Returns a list of valid index ranges for creating samples.

        Returns
        -------
        list
            List of valid indices for historical and prediction data extraction.
        """
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            # Only consider time between 6:00 and 24:00 for Metro data
            if "Metro" in self.data_name:
                if label_start_idx % (24 * 6) < (7 * 6):
                    continue
                if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                    continue

            recent = search_recent_data(
                self.feature, label_start_idx, self.T_p, self.T_h
            )  # recent data

            if recent:
                idx_lst.append(recent)

        return idx_lst
