# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Feng Wang
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in dl_lib.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use dl_lib as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import glob
import logging
import os
import re
import sys
sys.path.insert(0, '.')  # noqa: E402
from collections import OrderedDict

import dl_lib.utils.comm as comm
# from config_sam_multifea_finetune_2 import config
from config_sam_multifea_frozen import config
from dl_lib.checkpoint import DetectionCheckpointer
from dl_lib.data import MetadataCatalog
from dl_lib.engine import (DefaultTrainer, default_argument_parser,
                           default_setup, launch)
from dl_lib.evaluation import (COCOEvaluator, DatasetEvaluators,
                               PascalVOCDetectionEvaluator, verify_results)
# from net_sam_multifea import build_model
from net_sam_multifea_frozen import build_model

from dl_lib.data.datasets import register_coco_instances
# register_coco_instances("whu1024_train", {}, json_file="/media/disk3/lmy/dataset/whu1024/annotations/train.json",
#                         image_root="/media/disk3/lmy/dataset/whu1024/train/")
# register_coco_instances("whu1024_val", {}, json_file="/media/disk3/lmy/dataset/whu1024/annotations/train_val.json",
#                         image_root="/media/disk3/lmy/dataset/whu1024/train_val/")
# register_coco_instances("whu1024_val", {}, json_file="/media/disk3/lmy/dataset/whu1024/annotations/overlap_val_40.json",
#                         image_root="/media/disk3/lmy/dataset/whu1024/overlap_val_40/")  #test
# register_coco_instances("NWPU_val", {}, json_file="/media/disk3/lmy/dataset/NWPU/annotations/NWPU_instances_val.json",
#                         image_root="/media/disk3/lmy/dataset/NWPU/val_img/")
register_coco_instances("whu_mix_val", {}, json_file="/media/disk3/lmy/dataset/whu-mix/annotations/test-1-rebox.json",
                        image_root="/media/disk3/lmy/dataset/whu-mix/test-1/")
# register_coco_instances("whu_mix_train", {}, json_file="/media/disk3/lmy/dataset/whu-mix/annotations/train-rebox.json",
#                         image_root="/media/disk3/lmy/dataset/whu-mix/train/")
# 设置类别
from dl_lib.data import MetadataCatalog
# MetadataCatalog.get("whu1024_train").thing_classes = ['building']
# MetadataCatalog.get("whu1024_val").thing_classes = ['building']
# MetadataCatalog.get("NWPU_val").thing_classes = \
#     ['airplane', 'ship','storage_tank', 'baseball_diamond','tennis_court', 'basketball_court', 'ground_track_field','harbor', 'bridge', 'vehicle']
# MetadataCatalog.get("whu_mix_train").thing_classes = ['building']
MetadataCatalog.get("whu_mix_val").thing_classes = ['building']

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(
                COCOEvaluator(
                    dataset_name, cfg, True,
                    output_folder, dump=cfg.GLOBAL.DUMP_TEST
                ))

        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("dl_lib.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        from dl_lib.modeling import GeneralizedRCNNWithTTA
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def test_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--start-iter", type=int, default=0, help="start iter used to test")
    parser.add_argument("--end-iter", type=int, default=None,
                        help="end iter used to test")
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    return parser


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    if args.debug:
        batches = int(cfg.SOLVER.IMS_PER_BATCH / 8 * args.num_gpus)
        if cfg.SOLVER.IMS_PER_BATCH != batches:
            cfg.SOLVER.IMS_PER_BATCH = batches
            logger.warning("SOLVER.IMS_PER_BATCH is changed to {}".format(batches))

    if "MODEL.WEIGHTS" in args.opts:
        valid_files = [cfg.MODEL.WEIGHTS]
    else:
        list_of_files = glob.glob(os.path.join(cfg.OUTPUT_DIR, '*.pth'))
        assert list_of_files, "no pth file found in {}".format(cfg.OUTPUT_DIR)
        list_of_files.sort(key=os.path.getctime)
        latest_file = list_of_files[-1]
        if not args.end_iter:
            valid_files = [latest_file]
        else:
            files = [f for f in list_of_files if str(f) <= str(latest_file)]
            valid_files = []
            for f in files:
                try:
                    model_iter = int(re.split(r'(model_|\.pth)', f)[-3])
                except Exception:
                    logger.warning("remove {}".format(f))
                    continue
                if args.start_iter <= model_iter <= args.end_iter:
                    valid_files.append(f)
            assert valid_files, "No .pth files satisfy your requirement"

    # * means all if need specific format then *.csv
    for current_file in valid_files:
        cfg.MODEL.WEIGHTS = current_file
        model = build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))

    # return res


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
