"""
Vision Transformer bottleneck module for TransUNet.
Projects CNN feature maps into sequence tokens, applies multi-head self-attention,
and reshapes back to spatial feature maps.
"""
import torch
import torch.nn as nn

class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Projection from CNN channel dimension to Transformer embedding dimension
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection back to intermediate decoder dimension (e.g., 512 channels)
        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Project to embed_dim: [B, embed_dim, H, W]
        x_proj = self.proj_in(x)
        # Flatten spatial dimensions to sequence: [B, H*W, embed_dim]
        tokens = x_proj.flatten(2).permute(0, 2, 1)
        
        # Transformer attention
        tokens = self.transformer(tokens)
        
        # Reshape back to spatial feature map: [B, embed_dim, H, W]
        out_spatial = tokens.permute(0, 2, 1).view(B, self.embed_dim, H, W)
        return self.proj_out(out_spatial)
