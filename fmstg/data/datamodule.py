"""PyTorch Lightning Data Module Implementation for Traffic Data"""
from pathlib import Path

import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fmstg.data.dataset import (
    CleanDataset,
    CleanDatasetConfig,
    TrafficDataset,
    TrafficDatasetConfig,
)


class TrafficDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for traffic prediction tasks, which handles loading,
    splitting, and preparation of data for Spatio-Temporal Graph Convolutional Networks (ST-GCN).

    Attributes
    ----------
    data_name : str
        Name of the dataset (e.g., 'Metro', 'AIR').
    node_features_filepath : Path
        Path to the file containing the node features (e.g., traffic flow, weather data).
    adjacency_matrix_filepath : Path
        Path to the adjacency matrix representing spatial relationships between nodes.
    val_start_idx : int
        Starting index for validation data split.
    test_start_idx : int
        Starting index for test data split.
    alpha : Optional[int]
        Maximum number of hops for ST-GCN (optional).
    t_size : Optional[int]
        Size of the temporal window for ST-GCN (optional).
    T_h : int
        Number of historical time steps to use as input.
    T_p : int
        Number of future time steps to predict.
    V : int
        Number of nodes (e.g., sensors or stations).
    points_per_hour : int
        Number of data points collected per hour.
    train_batch_size : int
        Batch size for training.
    train_shuffle : bool
        Whether to shuffle the training data.
    """

    def __init__(self: "TrafficDataModule", config: DictConfig) -> None:
        """
        Initializes the TrafficDataModule by loading configurations for both clean and traffic datasets,
        validating file paths, and setting up batch sizes and data splits.

        Parameters
        ----------
        config : DictConfig
            Configuration object containing paths, indices, and other settings for the data.
        """
        super().__init__()
        #################### Clean Dataset ####################
        self.data_name = config.data.name
        self.node_features_filepath = Path(config.data.node_features_filepath)
        self.adjacency_matrix_filepath = Path(config.data.adjacency_matrix_filepath)

        # Deal with cluster issues
        self.prepare_data_per_node = False

        # Validate paths
        if not self.node_features_filepath.exists():
            raise FileExistsError(f"The file '{self.node_features_filepath}' does not exist!")
        if not self.adjacency_matrix_filepath.exists():
            raise FileExistsError(f"The file '{self.adjacency_matrix_filepath}' does not exist!")

        # Indices for data splits
        self.val_start_idx = config.data.val_start_idx
        self.test_start_idx = config.data.test_start_idx

        # Create clean dataset config
        self.clean_dataset_config = CleanDatasetConfig(
            data_name=self.data_name,
            node_features_filepath=str(self.node_features_filepath),
            adjacency_matrix_filepath=str(self.adjacency_matrix_filepath),
            val_start_idx=self.val_start_idx,
        )

        # For ST-GCN (Spatio-Temporal Graph Convolutional Networks)
        if config.model.get("alpha", None) is not None:
            self.alpha = config.alpha
            self.t_size = config.t_size

            self.clean_dataset_config.alpha = self.alpha
            self.clean_dataset_config.t_size = self.t_size

        #######################################################
        ################### Traffic Dataset ###################
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour

        self.traffic_dataset_config = TrafficDatasetConfig(
            T_h=self.T_h, T_p=self.T_p, V=self.V, points_per_hour=self.points_per_hour
        )
        #######################################################
        ################### Training Related ##################

        self.train_batch_size = config.data.train_batch_size
        self.train_shuffle = config.data.train_shuffle

        #######################################################

    def prepare_data(self: "TrafficDataModule") -> None:
        """Prepares the clean dataset by loading the actual data using CleanDataset.

        This method is typically called once during the setup phase, and it ensures that the data
        is loaded and ready for the TrafficDataset to wrap and split the data for different stages
        (train, validation, test).
        """
        self.clean_dataset = CleanDataset(self.clean_dataset_config)

    def setup(self: "TrafficDataModule", stage: str) -> None:
        """
        Sets up the train, validation, and test datasets using the TrafficDataset class, which wraps
        the CleanDataset and applies the necessary data splits based on the configuration.

        Parameters
        ----------
        stage : str
            The current stage of the training process ('fit', 'validate', 'test').
        """
        self.train_dataset = TrafficDataset(
            self.clean_dataset,
            # Use all the data from [0, val_start]
            (0 + self.T_p, self.val_start_idx - self.T_p + 1),
            self.traffic_dataset_config,
        )
        self.val_dataset = TrafficDataset(
            self.clean_dataset,
            # Use all the data from [val_start, test_start]
            (self.val_start_idx + self.T_p, self.test_start_idx - self.T_p + 1),
            self.traffic_dataset_config,
        )
        # Use all the data from [test_start, T_end]
        self.test_dataset = TrafficDataset(
            self.clean_dataset,
            (self.test_start_idx + self.T_p, -1),
            self.traffic_dataset_config,
        )

    def train_dataloader(self: "TrafficDataModule") -> DataLoader:
        """
        Returns a DataLoader for the training data.

        Returns
        -------
        DataLoader
            DataLoader for the training dataset with the specified batch size and shuffle settings.
        """
        return DataLoader(
            self.train_dataset,
            self.train_batch_size,
            shuffle=self.train_shuffle,
            pin_memory=True,
        )

    def val_dataloader(self: "TrafficDataModule") -> DataLoader:
        """
        Returns a DataLoader for the validation data.

        Returns
        -------
        DataLoader
            DataLoader for the validation dataset with a batch size of 64 and no shuffling.
        """
        return DataLoader(self.val_dataset, 64, shuffle=False, pin_memory=True)

    def test_dataloader(self: "TrafficDataModule") -> DataLoader:
        """
        Returns a DataLoader for the test data.

        Returns
        -------
        DataLoader
            DataLoader for the test dataset with a batch size of 64 and no shuffling.
        """
        return DataLoader(self.test_dataset, 64, shuffle=False, pin_memory=True)
