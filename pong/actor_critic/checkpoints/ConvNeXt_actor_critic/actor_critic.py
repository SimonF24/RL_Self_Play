import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """Modified from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"""
    def __init__(self, dim: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # Depthwise convolution
        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Linear(dim, 4 * dim) # Pointwise convolution 1 (expansion)
        self.activation = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim) # Pointwise convolution 2 (projection)
        
    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        # Permute from (C, H, W) to (H, W, C) for LayerNorm and Linear layers
        x = x.permute(1, 2, 0)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        # Permute back to (C, H, W)
        x = x.permute(2, 0, 1)
        return x + residual 


class PointwiseLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    
    Taken from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# This actor critic is very simple and something of a placeholder to be replaced later
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared encoder
        # The input is expected to be 4 stacked grayscale 210x160 images (shape [4, 210, 160])
        # This is shallow enough that we omit stochastic depth and layer scaling
        self.encoder = nn.Sequential(
            # Patchify stem a la ConvNeXt, inspired by ViT
            nn.Conv2d(4, 8, kernel_size=4, stride=4, padding=0), # Output: 4x52x40
            PointwiseLayerNorm(8, data_format="channels_first"),
            
            # First block
            ConvNeXtBlock(8),
            
            # Downsample
            PointwiseLayerNorm(8, data_format="channels_first"),
            nn.Conv2d(8, 32, kernel_size=2, stride=2), # Output: 32x26x20
            
            # Second block
            ConvNeXtBlock(32),
            
            # Downsample
            PointwiseLayerNorm(32, data_format="channels_first"),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), # Output: 64x13x10
            
            # Third block
            ConvNeXtBlock(64),

            # Global average pooling and flatten to get a 64-dimensional feature vector
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(0, -1) # We have no batch dimension so we flatten the entire output to get a 64-dimensional vector
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Softmax(dim=-1) # Output is a probability distribution over 6 actions
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output is a single value representing the state value
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        actor_output = self.actor(encoded)
        critic_output = self.critic(encoded)
        return actor_output.squeeze(), critic_output.squeeze()