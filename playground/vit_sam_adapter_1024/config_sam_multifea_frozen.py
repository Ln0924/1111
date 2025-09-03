import os.path as osp

from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/media/disk3/lmy/centernet-better-adapter-sam/checkpoint/sam_centernet_pretrained_whu_mix_0208.pth",
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
            # OUT_FEATURES=["res2", "res3", "res4", "res5"],
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
            TRAIN_NUM_POINTS=12544,
            OVERSAMPLE_RATIO=3.0,
            IMPORTANCE_SAMPLE_RATIO=0.75
        ),
        SAM = dict(
            EMBED_DIM=256,
            IMG_EMBEDDING_SIZE=(64, 64),
            INPUT_IMG_SIZE=(256, 256),
            TRANSFORMER_DIM=256,
        ),
        LOSS=dict(
            DICE_WEIGHT=1,
            MASK_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ('Resize',dict(shape=1024)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
                # ('Resize', dict(shape=1024)),
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(256, 256),
        # MASK_FORMAT="bitmask"
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    DATASETS=dict(
        TRAIN=("whu_mix_train",),
        TEST=("whu_mix_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            # NAME="SGD",  sgd是个坑，容易崩溃
            NAME="AdamW",
            BASE_LR=0.00001,
            WEIGHT_DECAY=0.05,
            BETAS=(0.9, 0.999),
            AMSGRAD=False,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            # STEPS=(4100, 5500),
            # MAX_ITER=7000,
            STEPS=(10000, 13000),
            MAX_ITER=15000,
            WARMUP_ITERS=10,
        ),
        IMS_PER_BATCH=4,
        CHECKPOINT_PERIOD=500,
    ),
    TEST=dict(
        EVAL_PERIOD=1000,
    ),
    OUTPUT_DIR=osp.join(
        '/media/disk3/lmy/centernet-better-adapter-sam/outputs/playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]+"_multifea_from_scrath_whu_mix_frozen_0213"
    ),
    GLOBAL=dict(DUMP_TEST=False),
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()
