"""
This module implements UGNet and various building blocks for spatio-temporal graph neural networks,
including temporal convolutional networks (TCN) and spatial blocks for graph convolution operations.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from cfmstg.utils.graph_prcs import asym_adj


class TimeEmbeddingLayer(nn.Module):
    """
    A layer that generates sinusoidal embeddings for timesteps.
    This matches the implementation in Denoising Diffusion Probabilistic Models.

    Attributes
    ----------
    embedding_dim : int
        Dimension of the embedding.
    """

    def __init__(self, embedding_dim: int):
        """
        Initializes the time embedding layer.

        Parameters
        ----------
        embedding_dim : int
            The dimension of the embedding.
        """
        super(TimeEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal embeddings for the input timesteps.

        Parameters
        ----------
        timesteps : torch.Tensor
            A tensor containing the timesteps (1D tensor).

        Returns
        -------
        torch.Tensor
            The sinusoidal time embeddings.
        """
        assert len(timesteps.shape) == 1

        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return emb


class SpatialBlock(nn.Module):
    """
    A block to extract spatial features using graph convolution operations.

    Attributes
    ----------
    theta : nn.Parameter
        A learnable weight matrix for spatial convolution.
    b : nn.Parameter
        A learnable bias parameter.
    """

    def __init__(self, ks: int, c_in: int, c_out: int):
        """
        Initializes the SpatialBlock.

        Parameters
        ----------
        ks : int
            The kernel size.
        c_in : int
            The number of input channels.
        c_out : int
            The number of output channels.
        """
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the parameters using Kaiming uniform initialization.
        """
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x: torch.Tensor, Lk: torch.Tensor) -> torch.Tensor:
        """
        Performs spatial convolution on the input data using the adjacency matrix.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch, c_in, time, n_nodes).
        Lk : torch.Tensor
            Graph adjacency matrix of shape (ks, n_nodes, n_nodes).

        Returns
        -------
        torch.Tensor
            The output after spatial convolution.
        """
        if len(Lk.shape) == 2:
            Lk = Lk.unsqueeze(0)

        # Perform graph convolution
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b

        return torch.relu(x_gc + x)


class Chomp(nn.Module):
    """
    A layer that removes the padding added during temporal convolutions.

    Attributes
    ----------
    chomp_size : int
        The amount of padding to remove.
    """

    def __init__(self, chomp_size: int):
        """
        Initializes the Chomp layer.

        Parameters
        ----------
        chomp_size : int
            The amount of padding to remove.
        """
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Removes the padding from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The tensor with padding removed.
        """
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    """
    A Temporal Convolutional Network (TCN) block for extracting temporal features.

    Attributes
    ----------
    conv : nn.Conv2d
        Convolutional layer for time series data.
    chomp : Chomp
        Chomp layer to remove padding.
    drop : nn.Dropout
        Dropout layer for regularization.
    net : nn.Sequential
        The sequential model that combines convolution, chomp, and dropout.
    shortcut : nn.Module
        A shortcut connection for residual learning.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        dilation_size: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initializes the TcnBlock.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        c_out : int
            The number of output channels.
        kernel_size : int
            The size of the kernel in temporal convolution.
        dilation_size : int, optional
            The dilation factor for temporal convolutions, by default 1.
        dropout : float, optional
            Dropout rate, by default 0.0.
        """
        super(TcnBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        # Temporal convolution layer
        self.conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(3, self.kernel_size),
            padding=(1, self.padding),
            dilation=(1, self.dilation_size),
        )

        self.chomp = Chomp(self.padding)
        self.drop = nn.Dropout(dropout)

        # Combine layers into a sequential model
        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        # Shortcut connection (identity if c_in == c_out, else a Conv2d layer)
        self.shortcut = (
            nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input through the TCN block and adds the residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T), where
            - V is the number of nodes
            - T is the number of time steps.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, c_out, V, T) with the residual connection.
        """
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)
        return out + x_skip


class ResidualBlock(nn.Module):
    """
    A residual block that combines temporal convolutions and graph convolutions.

    Attributes
    ----------
    tcn1 : TcnBlock
        The first TCN block.
    tcn2 : TcnBlock
        The second TCN block.
    shortcut : nn.Module
        Shortcut connection to preserve the original input.
    t_conv : nn.Conv2d
        A 1x1 convolution for time embedding.
    spatial : SpatialBlock
        A spatial block to extract spatial features.
    norm : nn.LayerNorm
        Layer normalization for the output.
    """

    def __init__(self, c_in: int, c_out: int, config, kernel_size: int = 3):
        """
        Initializes the ResidualBlock.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        c_out : int
            The number of output channels.
        config : Any
            Configuration object with graph and model settings.
        kernel_size : int, optional
            The size of the convolution kernel, by default 3.
        """
        super(ResidualBlock, self).__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = (
            nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1, 1))
        )
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1, 1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)
        self.norm = nn.LayerNorm([config.V, c_out])

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, A_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input through temporal and spatial convolutions with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor.
        A_hat : torch.Tensor
            Adjacency matrix for graph convolution.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, c_out, V, T) after residual connections.
        """
        h = self.tcn1(x)
        h += self.t_conv(t[:, :, None, None])
        h = self.tcn2(h)
        h = self.norm(h.transpose(1, 3)).transpose(1, 3)
        h = h.transpose(2, 3)
        h = self.spatial(h, A_hat).transpose(2, 3)
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    A down-sampling block in the U-Net architecture.

    Attributes
    ----------
    res : ResidualBlock
        The residual block used to extract temporal and spatial features during down-sampling.
    """

    def __init__(self, c_in: int, c_out: int, config):
        """
        Initializes the DownBlock.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        c_out : int
            The number of output channels.
        config : Any
            Configuration object with graph and model settings.
        """
        super(DownBlock, self).__init__()
        self.res = ResidualBlock(c_in, c_out, config, kernel_size=3)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor.
        supports : torch.Tensor
            Adjacency matrix for graph convolution.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, c_out, V, T).
        """
        return self.res(x, t, supports)


class Downsample(nn.Module):
    """
    A down-sampling layer that reduces the temporal resolution using convolution.

    Attributes
    ----------
    conv : nn.Conv2d
        A convolutional layer with stride (1, 2) to perform down-sampling.
    """

    def __init__(self, c_in: int):
        """
        Initializes the Downsample layer.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        """
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_in, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the down-sampling operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor (unused in this block).
        supports : torch.Tensor
            Adjacency matrix (unused in this block).

        Returns
        -------
        torch.Tensor
            Output tensor with down-sampled temporal resolution.
        """
        return self.conv(x)


class UpBlock(nn.Module):
    """
    An up-sampling block in the U-Net architecture.

    Attributes
    ----------
    res : ResidualBlock
        The residual block used to extract temporal and spatial features during up-sampling.
    """

    def __init__(self, c_in: int, c_out: int, config):
        """
        Initializes the UpBlock.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        c_out : int
            The number of output channels.
        config : Any
            Configuration object with graph and model settings.
        """
        super(UpBlock, self).__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, config, kernel_size=3)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor.
        supports : torch.Tensor
            Adjacency matrix for graph convolution.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, c_out, V, T).
        """
        return self.res(x, t, supports)


class Upsample(nn.Module):
    """
    An up-sampling layer that increases the temporal resolution using transposed convolution.

    Attributes
    ----------
    conv : nn.ConvTranspose2d
        A transposed convolutional layer to perform up-sampling.
    """

    def __init__(self, c_in: int):
        """
        Initializes the Upsample layer.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        """
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the up-sampling operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor (unused in this block).
        supports : torch.Tensor
            Adjacency matrix (unused in this block).

        Returns
        -------
        torch.Tensor
            Output tensor with up-sampled temporal resolution.
        """
        return self.conv(x)


class MiddleBlock(nn.Module):
    """
    The middle block in the U-Net architecture for refining features.

    Attributes
    ----------
    res1 : ResidualBlock
        The first residual block in the middle block.
    res2 : ResidualBlock
        The second residual block in the middle block.
    """

    def __init__(self, c_in: int, config):
        """
        Initializes the MiddleBlock.

        Parameters
        ----------
        c_in : int
            The number of input channels.
        config : Any
            Configuration object with graph and model settings.
        """
        super(MiddleBlock, self).__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in, config, kernel_size=3)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, supports: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input through two residual blocks for feature refinement.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, c_in, V, T).
        t : torch.Tensor
            Time embedding tensor.
        supports : torch.Tensor
            Adjacency matrix for graph convolution.

        Returns
        -------
        torch.Tensor
            Output tensor after two residual blocks.
        """
        x = self.res1(x, t, supports)
        x = self.res2(x, t, supports)
        return x


class UGnet(nn.Module):
    """
    A U-Net based graph neural network for spatio-temporal tasks.

    Attributes
    ----------
    config : Any
        Configuration object with graph and model settings.
    down : nn.ModuleList
        A list of down-sampling blocks.
    middle : MiddleBlock
        The middle block of the U-Net.
    up : nn.ModuleList
        A list of up-sampling blocks.
    x_proj : nn.Conv2d
        A projection layer to match input channels.
    out : nn.Sequential
        Output layer that generates the final prediction.
    a1 : torch.Tensor
        Precomputed adjacency matrix for graph convolution (asymmetrically normalized).
    a2 : torch.Tensor
        Transposed adjacency matrix for reverse graph convolution.
    """

    def __init__(self, config):
        """
        Initializes the UGnet model.

        Parameters
        ----------
        config : Any
            Configuration object with graph and model settings.
        """
        super(UGnet, self).__init__()
        self.config = config
        self.d_h = config.d_h
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = self.T_p + self.T_h
        self.F = config.F

        self.n_blocks = config.get("n_blocks", 2)
        n_resolutions = len(config.channel_multipliers)

        # Build down-sampling path
        down = []
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, config)

        # Build up-sampling path
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, config))
            out_channels = in_channels // config.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, config))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        # Input and output projection layers
        self.x_proj = nn.Conv2d(self.F, self.d_h, (1, 1))
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, self.F, (1, 1)),
            nn.Linear(2 * T, T),
        )

        # Graph convolution adjacency matrices
        a1 = asym_adj(np.load(config.adj_matrix_filepath))
        a2 = asym_adj(np.transpose(np.load(config.adj_matrix_filepath)))
        self.a1 = torch.from_numpy(a1).to(torch.device(config.device_name))
        self.a2 = torch.from_numpy(a2).to(torch.device(config.device_name))
        config.supports_len = 2

        # Time Embedding
        self.time_ebedding_layer = TimeEmbeddingLayer(self.d_h)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input through the U-Net with down-sampling, middle block, and up-sampling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, F, V, T), where
            - F is the number of features
            - V is the number of nodes
            - T is the number of time steps.
        t : torch.Tensor
            Time embedding tensor.
        c : torch.Tensor
            Condition tensor containing x_masked, pos_w, and pos_d (additional information for the model).

        Returns
        -------
        torch.Tensor
            The final output tensor after U-Net processing.
        """
        x_masked, _, _ = c
        print(f"Forward arguments: x: {x.shape}, x_masked: {x_masked.shape}")
        x = torch.cat((x, x_masked), dim=3)  # Concatenate with masked data
        print(f"Model input: {x.shape}")
        x = self.x_proj(x)

        t = self.time_ebedding_layer(t)  # Apply time embedding
        h = [x]

        supports = torch.stack([self.a1, self.a2])

        # Down-sampling path
        for m in self.down:
            x = m(x, t, supports)
            h.append(x)

        # Middle block
        x = self.middle(x, t, supports)

        # Up-sampling path
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, supports)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, supports)

        e = self.out(x)

        print(f"Model output: {e.shape}")
        return e