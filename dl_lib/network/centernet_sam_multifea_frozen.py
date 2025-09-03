import math

import numpy as np
import torch
import torch.nn as nn
from ..utils.comm import get_world_size
from ..utils.events import get_event_storage
from torch.nn import functional as F
from ..utils.memory import retry_if_cuda_oom
from dl_lib.layers import ShapeSpec
from dl_lib.structures import Boxes, ImageList, Instances

from .generator import CenterNetDecoder, CenterNetGT
from .loss import modified_focal_loss, reg_l1_loss, HungarianMatcher, sigmoid_ce_loss_jit, dice_loss_jit
from ..utils.misc import nested_tensor_from_tensor_list,is_dist_avail_and_initialized
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess

class CenterNet_Sam_multifea_frozen(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.upsample = cfg.build_upsample_layers(cfg)
        self.head = cfg.build_head(cfg)
        self.prompt_encoder = cfg.build_promptencoder(cfg)
        self.mask_decoder = cfg.build_maskdecoder(cfg)
        # self.cls_head = cfg.build_cls_head(cfg)
        # self.wh_head = cfg.build_width_height_head(cfg)
        # self.reg_head = cfg.build_center_reg_head(cfg)

        # backbone_shape = self.backbone.output_shape()
        # feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.oversample_ratio = cfg.MODEL.CENTERNET.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.CENTERNET.IMPORTANCE_SAMPLE_RATIO
        self.mask_weight = cfg.MODEL.LOSS.MASK_WEIGHT
        self.dice_weight = cfg.MODEL.LOSS.DICE_WEIGHT
        self.num_points = cfg.MODEL.CENTERNET.TRAIN_NUM_POINTS
        self.matcher = HungarianMatcher(
            cost_mask=self.mask_weight,
            cost_dice=self.dice_weight,
            num_points=self.num_points,
        )

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(self.mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                batched_inputs:{"image":(B,C,H,W),"instances":instances}
                instances.gt_boxes instances.gt_masks instances.gt_classes

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        if not self.training:
            images = self.preprocess_image_inference(batched_inputs)
            return self.inference(images)
        # training
        images = self.preprocess_image(batched_inputs)
        # print("input_images",images.image_sizes)

        # current_iter = get_event_storage().iter
        # print(current_iter)
        with torch.no_grad():
            features = self.backbone(images.tensor)
            up_fmap = self.upsample(features)
            pred_box_dict = self.head(up_fmap)  # pred = {'cls': cls,'wh': wh,'reg': reg,}

        # gt_box_dict = self.get_ground_truth_box(batched_inputs)  #gt_dict = {"score_map","wh","reg","reg_mask","index",}

        # if current_iter < self.cfg.SOLVER.PARTIAL_ITERATIONS:
        #     return self.losses_bcox(pred_box_dict, gt_box_dict)

        # get the decode box
        # print(features.key)
        # no detach
        adapter_feature_256 = features["res2"]
        adapter_feature_128 = features["res3"]
        # adapter_feature_64 = features["res4"]

        # detach
        vit_output = features["x4"].clone().detach()  #batch,256,64,64
        hm = pred_box_dict["cls"].clone().detach()
        wh = pred_box_dict["wh"].clone().detach()
        reg = pred_box_dict["reg"].clone().detach()
       # fm = pred_box_dict["fm"].clone().detach()
        bboxes, scores, clses, center_points = CenterNetDecoder.decode(hm, wh, reg)  #256大小上的，4倍降采样
        # print("center_points",center_points.shape)
        # print("bboxes",bboxes.shape)
        # bboxes(batch,k,4),center_points (batch,k,2)需要变形
        batch_num = bboxes.shape[0]
        batch_res_masks_256 = []
        batch_res_masks_128 = []
        # batch_res_masks_64 = []

        for i in range(batch_num):
            bboxes_per_img = bboxes[i]  #k,4
            center_points_coord_per_img = center_points[i]  #k,2
            # print("bboxes_per_img",bboxes_per_img.shape)
            # bboxes_per_img = bboxes_per_img.unsqueeze(1)  #k,1,4
            center_points_coord_per_img = center_points_coord_per_img.unsqueeze(1)  #k,1,2
            # print("center_points_coord_per_img",center_points_coord_per_img.shape)
            center_points_labels = torch.ones([center_points_coord_per_img.shape[0],center_points_coord_per_img.shape[1]]).to(self.device)  #k,1
            # print("center_points_labels",center_points_labels.shape)
            center_points_per_img = (center_points_coord_per_img, center_points_labels)

            vit_output_per_img = vit_output[i]  #256,64,64
            adapter_feature_256_per_img = adapter_feature_256[i]
            adapter_feature_128_per_img = adapter_feature_128[i]
            # adapter_feature_64_per_img = adapter_feature_64[i]

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=center_points_per_img,
                    boxes=bboxes_per_img,
                    masks=None
                )
            #     curr_embedding 从哪里取
            res_masks = self.mask_decoder(
                image_embeddings=vit_output_per_img.unsqueeze(0),  # C,h_encoder,w_encoder-->1,C,h_encoder,w_encoder
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                adapter_feature1=adapter_feature_256_per_img.unsqueeze(0),
                adapter_feature2=adapter_feature_128_per_img.unsqueeze(0),
                # adapter_feature3=adapter_feature_64_per_img.unsqueeze(0),
            )  #"res_masks_nums": k,1,nums,nums
            # print("low_res_masks",low_res_masks.shape)
            res_masks_256 = res_masks["mask_256"].squeeze(1)
            res_masks_128 = res_masks["mask_128"].squeeze(1)
            # res_masks_64 = res_masks["mask_64"].squeeze(1)
            # res_masks = res_masks.squeeze(1)  #k,256,256
            batch_res_masks_256.append(res_masks_256)
            batch_res_masks_128.append(res_masks_128)
            # batch_res_masks_64.append(res_masks_64)

        batch_res_masks_256 = torch.stack([mask for mask in batch_res_masks_256], dim=0)
        batch_res_masks_128 = torch.stack([mask for mask in batch_res_masks_128], dim=0)
        # batch_res_masks_64 = torch.stack([mask for mask in batch_res_masks_64], dim=0)

        # batch_low_res_masks = torch.tensor(batch_low_res_masks)  #batch,k,256,256
        # print("batch_low_res_masks",batch_low_res_masks.shape)
        # pred_mask_dict={"outputs":[{"pred_masks":batch_res_masks_256},
        #                 {"pred_masks":batch_res_masks_128},
        #                 {"pred_masks":batch_res_masks_64}]
        #                 }
        pred_mask_dict = {"outputs": [{"pred_masks": batch_res_masks_256},
                                      {"pred_masks": batch_res_masks_128}]
                          }
        gt_mask_dict = self.get_ground_truth_mask(batched_inputs,images)  #batch,N,1024,1024 gt_mask_dict = {"masks"}


        return self.losses(pred_mask_dict, gt_mask_dict)


    def losses(self, pred_mask_dict, gt_mask_dict):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "wh": gt width and height of boxes,
                "reg": gt regression of box center point,
                "reg_mask": mask of regression,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "cls": predicted score map
            "reg": predcited regression
            "wh": predicted width and height of box
        }
        """
        losses = {}
        # loss for masks in sam
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["masks"]) for t in gt_mask_dict)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=self.device)
        #  不确定这里要不要对不对
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        for i, output in enumerate(pred_mask_dict["outputs"]):
            # matcher的过程会算100个mask和N个gt的匹配，然后实际算损失的时候，只有N个匹配上的是参与计算的
            indices = self.matcher(output, gt_mask_dict)
            # Compute all the requested losses
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            src_masks = output["pred_masks"]
            src_masks = src_masks[src_idx]
            masks = [t["masks"] for t in gt_mask_dict]
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)
            target_masks = target_masks[tgt_idx]

            # No need to upsample predictions as we are using normalized coordinates :)
            # N x 1 x H x W
            src_masks = src_masks[:, None]
            target_masks = target_masks[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: self.calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
            loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)
            loss_mask *= self.cfg.MODEL.LOSS.MASK_WEIGHT
            loss_dice *= self.cfg.MODEL.LOSS.DICE_WEIGHT
            loss = {f"loss_mask_{i}":loss_mask, f"loss_dice_{i}":loss_dice}
            losses.update(loss)

        # loss = {
        #     "loss_cls": loss_cls,
        #     "loss_box_wh": loss_wh,
        #     "loss_center_reg": loss_reg,
        #     "loss_dice":loss_dice,
        #     "loss_mask":loss_mask
        # }
        # print(list(losses.keys()))
        # [ 'loss_mask_0', 'loss_dice_0', 'loss_mask_1', 'loss_dice_1', 'loss_mask_2', 'loss_dice_2']
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_uncertainty(self,logits):
        """
        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
            foreground class in `classes`.
        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))

    @torch.no_grad()
    def get_ground_truth_box(self, batched_inputs):
        return CenterNetGT.generate(self.cfg, batched_inputs)

    @torch.no_grad()
    def get_ground_truth_mask(self,batched_inputs,images):
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        gt_mask_dict = self.prepare_mask_targets(gt_instances, images)
        return gt_mask_dict


    @torch.no_grad()
    def inference(self, images):
        """
        image(tensor): ImageList in dl_lib.structures
        """
        n, c, h, w = images.tensor.shape
        # print("images.tensor.shape",images.tensor.shape)
        need_h, need_w = 1024, 1024
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([w, h], dtype=np.float32)
        # size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        # img_info = dict(center=center_wh, size=size_wh,
        #                 height=h // down_scale,
        #                 width=w // down_scale)
        img_info = dict(center=center_wh, size=size_wh,
                        height=256,
                        width=256)

        # features = self.backbone(aligned_img)
        if h != need_h or w != need_w:
            input_images = F.interpolate(images.tensor, size=(need_h, need_w), mode="bilinear", align_corners=False)
        else:
            input_images = images.tensor
        # print("input_images",input_images.shape)
        features = self.backbone(input_images)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)
        # results = self.decode_prediction(pred_dict, img_info)
        adapter_feature_256 = features["res2"]
        adapter_feature_128 = features["res3"]
        # adapter_feature_64 = features["res4"]
        vit_output = features["x4"]
        fmap = pred_dict["cls"]
        reg = pred_dict["reg"]
        wh = pred_dict["wh"]

        boxes, scores, classes, center_points = CenterNetDecoder.decode(fmap, wh, reg)
        scores = scores.reshape(-1)  #B,K,1-->B*K*1
        classes = classes.reshape(-1).to(torch.int64)

        # dets = CenterNetDecoder.decode(fmap, wh, reg)
        boxes_de = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes_de = Boxes(boxes_de)

        batch_num = boxes.shape[0]
        batch_res_masks_256 = []
        batch_res_masks_128 = []
        # batch_res_masks_64 = []

        for i in range(batch_num):
            bboxes_per_img = boxes[i]  # k,4
            center_points_coord_per_img = center_points[i]  # k,2
            # print("bboxes_per_img",bboxes_per_img.shape)
            # bboxes_per_img = bboxes_per_img.unsqueeze(1)  #k,1,4
            center_points_coord_per_img = center_points_coord_per_img.unsqueeze(1)  # k,1,2
            # print("center_points_coord_per_img",center_points_coord_per_img.shape)
            center_points_labels = torch.ones(
                [center_points_coord_per_img.shape[0], center_points_coord_per_img.shape[1]]).to(self.device)  # k,1
            # print("center_points_labels",center_points_labels.shape)
            center_points_per_img = (center_points_coord_per_img, center_points_labels)

            vit_output_per_img = vit_output[i]  # 256,64,64
            adapter_feature_256_per_img = adapter_feature_256[i]
            adapter_feature_128_per_img = adapter_feature_128[i]
            # adapter_feature_64_per_img = adapter_feature_64[i]

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=center_points_per_img,
                    boxes=bboxes_per_img,
                    masks=None
                )
            #     curr_embedding 从哪里取
            res_masks = self.mask_decoder(
                image_embeddings=vit_output_per_img.unsqueeze(0),  # C,h_encoder,w_encoder-->1,C,h_encoder,w_encoder
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                adapter_feature1=adapter_feature_256_per_img.unsqueeze(0),
                adapter_feature2=adapter_feature_128_per_img.unsqueeze(0),
                # adapter_feature3=adapter_feature_64_per_img.unsqueeze(0),
            )  # "res_masks_nums": k,1,nums,nums
            # print("low_res_masks",low_res_masks.shape)
            res_masks_256 = res_masks["mask_256"].squeeze(1)
            res_masks_128 = res_masks["mask_128"].squeeze(1)
            # res_masks_64 = res_masks["mask_64"].squeeze(1)
            # res_masks = res_masks.squeeze(1)  #k,256,256
            batch_res_masks_256.append(res_masks_256)
            batch_res_masks_128.append(res_masks_128)
            # batch_res_masks_64.append(res_masks_64)

        batch_res_masks_256 = torch.stack([mask for mask in batch_res_masks_256], dim=0)
        batch_res_masks_128 = torch.stack([mask for mask in batch_res_masks_128], dim=0)
        # batch_res_masks_64 = torch.stack([mask for mask in batch_res_masks_64], dim=0)
        mask_pred_results_256 = F.interpolate(
            batch_res_masks_256,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  #b,K,1024,1024
        # print("batch_res_masks_128",batch_res_masks_128.shape)

        mask_pred_results_128 = F.interpolate(
            batch_res_masks_128,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  # b,K,1024,1024
        # print("mask_pred_results_128",mask_pred_results_128.shape)
        # mask_pred_results_64 = F.interpolate(
        #     batch_res_masks_64,
        #     size=(h, w),
        #     mode="bilinear",
        #     align_corners=False,
        # )  # b,K,1024,1024
        # mask_pred_results = (mask_pred_results_64+mask_pred_results_128+mask_pred_results_256)/3
        mask_pred_results = (mask_pred_results_128 + mask_pred_results_256) / 2
        # mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
        #     mask_pred_result, image_size, height, width)
        mask_pred_results = mask_pred_results.reshape(-1,h,w) #b*k,h,w
        mask_results =(mask_pred_results>0).float()
        ori_w, ori_h = w, h
        result = Instances((int(ori_h), int(ori_w)))
        result.pred_masks = mask_results
        result.pred_boxes = boxes_de

        mask_scores_per_image = (mask_pred_results.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores * mask_scores_per_image
        result.pred_classes = classes

        # det_instance = Instances((int(ori_h), int(ori_w)), **results)

        return [{"instances": result}]

    # def decode_prediction(self, pred_dict, img_info):
    #     """
    #     Args:
    #         pred_dict(dict): a dict contains all information of prediction
    #         img_info(dict): a dict contains needed information of origin image
    #     """
    #     fmap = pred_dict["cls"]
    #     reg = pred_dict["reg"]
    #     wh = pred_dict["wh"]
    #
    #     boxes, scores, classes, center_points = CenterNetDecoder.decode(fmap, wh, reg)
    #     # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
    #     scores = scores.reshape(-1)
    #     classes = classes.reshape(-1).to(torch.int64)
    #
    #     # dets = CenterNetDecoder.decode(fmap, wh, reg)
    #     boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
    #     boxes = Boxes(boxes)
    #     return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_image_inference(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255) for img in images]
        images = ImageList.from_tensors(images, 0)
        return images

    def prepare_mask_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "masks": padded_masks,
                }
            )
        return new_targets
