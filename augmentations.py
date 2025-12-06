# Chris
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")

class TimeMasking(nn.Module):
    """

    Mask time steps from every sample in the training set


    Input shape:
        (timesteps, channels) OR (batch, timesteps, channels)

    Args:
        max_mask_len: maximum number of timesteps to mask
        p: probability of applying masking to each sample
    """
    
    def __init__(self, max_mask_len=40, p=0.5):
        super().__init__()
        self.max_mask_len = max_mask_len
        self.p = p

    def forward(self, x):
        """
        [timesteps, channels] or [batch, timesteps, channels]
        dataset.py--> neural_feats
        """

        if x.dim() == 2:
            timesteps, channels = x.shape

            if torch.rand(1).item() < self.p:
                mask_len = torch.randint(1, self.max_mask_len + 1, (1,)).item()
                start = torch.randint(0, max(1, timesteps - mask_len), (1,)).item()
                x[start:start + mask_len, :] = 0
            return x


        batch, timesteps, channels = x.shape
        for b in range(batch):
            if torch.rand(1).item() < self.p:
                mask_len = torch.randint(1, self.max_mask_len + 1, (1,)).item()
                start = torch.randint(0, max(1, timesteps - mask_len), (1,)).item()
                x[b, start:start + mask_len, :] = 0

        return x

class FeatureMasking(nn.Module):
    """
    like time masking but modifies y-axis
    [timesteps, channels] or [batch, timesteps, channels]
        dataset.py--> neural_feats
    """
    def __init__(self, max_features=10, p=0.5):
        super().__init__()
        self.max_features = max_features
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x

        if x.dim() == 2:
            timesteps, channels = x.shape
            batch = None
        else:
            batch, timesteps, channels = x.shape

        # pick how many channels to mask
        k = torch.randint(1, self.max_features + 1, (1,)).item()

        # pick which channels to zero out
        channels = torch.randperm(channels)[:k]

        if batch is None:
            x[:, channels] = 0
        else:
            x[:, :, channels] = 0

        return x
