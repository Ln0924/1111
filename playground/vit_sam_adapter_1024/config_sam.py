import os.path as osp

from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-18.pth",
        WEIGHTS="/media/disk3/lmy/centernet-better-adapter-sam/checkpoint/sam_centernet_pretrained_nwpu1024.pth",
        MASK_ON=True,
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
            OUT_FEATURES=["res2", "res3", "res4", "res5","x4"],
            CONVS_DIM=256,
        ),
        PIXEL_MEAN=[0.485, 0.456, 0.406],
        PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            DECONV_CHANNEL=[1280, 1280, 1280, 1280, 1280, 64],
            DECONV_KERNEL=[4, 4, 4, 4],
            NUM_CLASSES=10,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=256,
            TRAIN_NUM_POINTS=12544,
            OVERSAMPLE_RATIO=3.0,
            IMPORTANCE_SAMPLE_RATIO=0.75
        ),
        SAM = dict(
            EMBED_DIM=256,
            IMG_EMBEDDING_SIZE=(64, 64),
            INPUT_IMG_SIZE=(256, 256),  #与centernet出来的4倍降采样框和点相匹配
            TRANSFORMER_DIM=256,
        ),
        LOSS=dict(
            CLS_WEIGHT=0.5,
            WH_WEIGHT=0.05,
            REG_WEIGHT=0.5,
            DICE_WEIGHT=1,
            MASK_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                # ('CenterAffine', dict(
                #     boarder=128,
                #     output_size=(1024, 1024),
                #     random_aug=True)),
                ('Resize',dict(shape=1024)),
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
        TRAIN=("NWPU_train",),
        TEST=("NWPU_val",),
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
            STEPS=(2000, 3000),
            MAX_ITER=4000,
            WARMUP_ITERS=100,
        ),
        IMS_PER_BATCH=4,
        CHECKPOINT_PERIOD=200,
    ),
    TEST=dict(
        EVAL_PERIOD=200,
    ),
    # OUTPUT_DIR=osp.join(
    #     '/data/Outputs/model_logs/playground',
    #     osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    # ),
    OUTPUT_DIR=osp.join(
        '/media/disk3/lmy/centernet-better-adapter-sam/outputs/playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]+"_box_0.5_2000_3000_4000_NWPU"
    ),
    GLOBAL=dict(DUMP_TEST=False),
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()
