# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from .transformer import TwoWayTransformer
from typing import List, Tuple, Type
from dl_lib.layers.batch_norm import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        cfg,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super(MaskDecoder,self).__init__()
        self.transformer_dim = cfg.MODEL.SAM.TRANSFORMER_DIM
        self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=cfg.MODEL.SAM.TRANSFORMER_DIM,
                mlp_dim=2048,
                num_heads=8,
            )

        # self.iou_token = nn.Embedding(1, self.transformer_dim)  #1,256
        # self.num_mask_tokens = num_multimask_outputs + 1
        self.num_mask_tokens = 1
        self.mask_token = nn.Embedding(self.num_mask_tokens, self.transformer_dim)  # 1,256

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlp = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # self.iou_prediction_head = MLP(
        #     self.transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        # if multimask_output:
        #     mask_slice = slice(1, None)  # whole, part, subpart
        # else:
        #     mask_slice = slice(0, 1)  # 多prompt的情况才输出0通道的无歧义的mask
        mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = self.mask_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  #n_points,1,256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  #n_points,3,256

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)  #n_points,256,64,64
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # print(src.shape)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # hs:n_points,7,256 src:n_points,256,64,64
        # iou_token_out = hs[:, 0, :]  #n_points,256
        # mask_tokens_out = hs[:, 1, :]  #n_points,4,256
        mask_tokens_out = hs[:, 0, :]  #100,256

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)  #n_points,32,256,256
        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)  #n_points,4,32
        hyper_in = self.output_hypernetworks_mlp[0](mask_tokens_out)  #100,32
        hyper_in = hyper_in.unsqueeze(1)
        # print(hyper_in.shape)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)   #n_points,4,256,256
        # print(masks.shape)

        # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)  #n_points,4

        # return masks, iou_pred
        return masks


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
