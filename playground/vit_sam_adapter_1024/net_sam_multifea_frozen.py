from dl_lib.network.backbone import Backbone
from dl_lib.layers import ShapeSpec
# from dl_lib.network import ResnetBackbone
# from dl_lib.network import D2ImageEncoderViT
from dl_lib.network import D2ImageEncoderViTAdapter
from dl_lib.network import CenternetDeconv1
from dl_lib.network import CenternetHead
from dl_lib.network import CenterNet_Sam_multifea_frozen
from dl_lib.network import PromptEncoder
from dl_lib.network import MaskDecoder_multifea
from dl_lib.network import MaskDecoder


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = D2ImageEncoderViTAdapter(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_upsample_layers(cfg, ):
    upsample = CenternetDeconv1(cfg)
    return upsample


def build_head(cfg, ):
    head = CenternetHead(cfg)
    return head

def build_promptencoder(cfg,):
    prompt_encoder = PromptEncoder(cfg)
    return prompt_encoder

def build_maskdecoder(cfg,):
    maskdecoder = MaskDecoder_multifea(cfg)
    return maskdecoder


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_upsample_layers = build_upsample_layers
    cfg.build_head = build_head
    cfg.build_promptencoder = build_promptencoder
    cfg.build_maskdecoder = build_maskdecoder
    model = CenterNet_Sam_multifea_frozen(cfg)
    return model
