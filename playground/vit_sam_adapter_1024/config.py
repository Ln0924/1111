import os.path as osp

from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-18.pth",
        WEIGHTS="/media/disk3/lmy/mask2former/checkpoint/sam_vit_h_4b8939_rename.pth",
        MASK_ON=False,
        # RESNETS=dict(DEPTH=50),
        VIT=dict(
            IMAGE_SIZE=1024,
            EMBED_DIM=1280,
            VIT_PATCH_SIZE=16,
            DEPTH=32,
            NUM_HEAD=16,
            WINDOW_SIZE=14,
            ENCODER_GLOBAL_ATTN_INDEXES=[7, 15, 23, 31],
            FROZEN_STAGES=32,
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
            CONVS_DIM=256,
        ),
        PIXEL_MEAN=[0.485, 0.456, 0.406],
        PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            DECONV_CHANNEL=[1280, 1280, 1280, 1280, 1280, 64],
            DECONV_KERNEL=[4, 4, 4, 4],
            NUM_CLASSES=1,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=256,
        ),
        LOSS=dict(
            CLS_WEIGHT=1,
            WH_WEIGHT=0.1,
            REG_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                # ('CenterAffine', dict(
                #     boarder=128,
                #     output_size=(1024, 1024),
                #     random_aug=True)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(256, 256),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    DATASETS=dict(
        TRAIN=("whu1024_train",),
        TEST=("whu1024_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            # NAME="SGD",  sgd是个坑，容易崩溃
            NAME="AdamW",
            BASE_LR=0.0001,
            WEIGHT_DECAY=0.05,
            BETAS=(0.9, 0.999),
            AMSGRAD=False,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(19000, 22000),
            MAX_ITER=24000,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=10,
    ),
    # OUTPUT_DIR=osp.join(
    #     '/data/Outputs/model_logs/playground',
    #     osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    # ),
    OUTPUT_DIR=osp.join(
        '/media/disk3/lmy/centernet-better-adapter/outputs/playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
    GLOBAL=dict(DUMP_TEST=False),
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()
