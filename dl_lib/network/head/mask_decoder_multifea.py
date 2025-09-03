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



class MaskDecoder_multifea(nn.Module):
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
        super(MaskDecoder_multifea,self).__init__()
        self.transformer_dim = cfg.MODEL.SAM.TRANSFORMER_DIM
        self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=cfg.MODEL.SAM.TRANSFORMER_DIM,
                mlp_dim=2048,
                num_heads=8,
            )

        # self.iou_token = nn.Embedding(1, self.transformer_dim)  #1,256
        # self.num_mask_tokens = num_multimask_outputs + 1
        self.num_mask_tokens = 2  #for 3 feature scale except 32
        self.mask_token = nn.Embedding(self.num_mask_tokens, self.transformer_dim)  # 3,256

        self.output_upscaling1 = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )  #64-->256  for layer1 256
        self.output_upscaling2 = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            activation(),
            nn.Conv2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=3, stride=1, padding=1,
                      bias=False),
            activation(),
        )  #64-->128  for layer2 128


        self.output_hypernetworks_mlp = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.adapter_feature_channel = cfg.MODEL.CENTERNET.DECONV_CHANNEL[:3]
        # only compress the channel not the size
        self.compress_vit_feat1 = nn.Sequential(
            nn.Conv2d(self.adapter_feature_channel[0], self.transformer_dim,kernel_size=3, stride=1,padding=1, bias=False),
            LayerNorm2d(self.transformer_dim),
            nn.GELU(),
            nn.Conv2d(self.transformer_dim, self.transformer_dim // 8,kernel_size=3, stride=1,padding=1, bias=False))  # 1280-->256-->32  for layer1 256
        self.compress_vit_feat2 = nn.Sequential(
            nn.Conv2d(self.adapter_feature_channel[1], self.transformer_dim, kernel_size=3, stride=1, padding=1,
                      bias=False),
            LayerNorm2d(self.transformer_dim),
            nn.GELU(),
            nn.Conv2d(self.transformer_dim, self.transformer_dim // 8, kernel_size=3, stride=1, padding=1,
                      bias=False))  # 1280-->256-->32    for layer2 128


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_feature1: torch.Tensor,
        adapter_feature2: torch.Tensor,
        # adapter_feature3: torch.Tensor,
    ):
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
        adapter_feature1 = self.compress_vit_feat1(adapter_feature1)  #1,32,256,256
        adapter_feature2 = self.compress_vit_feat2(adapter_feature2)  #1,32,128,128
        # adapter_feature3 = self.compress_vit_feat3(adapter_feature3)  #1,32,64,64
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            adapter_feature1=adapter_feature1,
            adapter_feature2=adapter_feature2,
            # adapter_feature3=adapter_feature3
        )

        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_feature1: torch.Tensor,
        adapter_feature2: torch.Tensor,
        # adapter_feature3: torch.Tensor
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  #n_points,3,256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  #n_points,3+2,256

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)  #n_points,256,64,64

        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # print(src.shape)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # hs:n_points,7,256 src:n_points,256,64,64

        mask_tokens_out = hs[:, 0:self.num_mask_tokens, :]  #100,3,256


        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding1 = self.output_upscaling1(src)+adapter_feature1.repeat(b,1,1,1)   #n_points,32,256,256
        upscaled_embedding2 = self.output_upscaling2(src)+adapter_feature2.repeat(b,1,1,1)   #n_points,32,128,128


        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlp[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  #n_points,3,32

        b1, c1, h1, w1 = upscaled_embedding1.shape

        masks_256 = ((hyper_in[:,0].unsqueeze(1)) @ upscaled_embedding1.view(b1, c1, h1 * w1)).view(b1, -1, h1, w1)   #n_points,1,256,256


        b2, c2, h2, w2 = upscaled_embedding2.shape
        masks_128 = ((hyper_in[:,1].unsqueeze(1)) @ upscaled_embedding2.view(b2, c2, h2 * w2)).view(b2, -1, h2, w2)


        masks = {"mask_256":masks_256,"mask_128":masks_128}
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
