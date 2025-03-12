# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling.

    This custom Vision Transformer model allows for the option of using global
    average pooling (instead of the default CLS token pooling) to aggregate
    features for classification.
    """
    def __init__(self, global_pool=False, **kwargs):
        # Call the parent constructor to initialize the base model
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        # If global pooling is enabled, add normalization after pooling
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # Remove the original normalization layer as we are replacing it with fc_norm
            del self.norm

    def forward_features(self, x):
        """
        Forward pass to extract features before classification head.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output features, either from CLS token or global average pooling.
        """
        B = x.shape[0]
        # Apply patch embedding
        x = self.patch_embed(x)

        # Add CLS token to the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply global average pooling or CLS token pooling
        if self.global_pool:
            # Global average pooling (ignores CLS token)
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            # Use the CLS token for classification
            x = self.norm(x)
            x = x[:, 0]

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """
        Forward pass through the head (classification layer).

        Args:
            x (Tensor): Input feature tensor.
            pre_logits (bool): If True, return the features before the classification head.

        Returns:
            Tensor: Output tensor after passing through the head.
        """
        # Apply normalization and dropout
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def vit_large_patch16(**kwargs):
    """
    Create a Vision Transformer (ViT) large model with patch size 16.

    Args:
        **kwargs: Additional arguments to customize the model.

    Returns:
        VisionTransformer: A Vision Transformer model with large configuration.
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
