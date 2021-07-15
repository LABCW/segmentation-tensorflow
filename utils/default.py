import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME ='DeepLabV3_plus'
_C.MODEL.FRONTEND ='ResNet50'
_C.MODEL.WIDTH =768
_C.MODEL.HEIGHT =768
_C.MODEL.BATCH = 1
_C.MODEL.CHANNEL_NUM = 3
_C.MODEL.class_model = 'multi_label'

_C.DATASET = CN()
_C.DATASET.ROOT = './level1_training_data/'
_C.DATASET.image_train_dir = 'ng'
_C.DATASET.mask_train_dir = 'labels'
_C.DATASET.mask_suffix = '_mask.png'
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.CLASSES_NAME = ['ng', 'pt']
_C.DATASET.ok_image_dir = None

_C.TRAIN = CN()
_C.TRAIN.checkpoint_path = 'level1_ckpt'
_C.TRAIN.num_epochs = 10000
_C.TRAIN.epoch_start_i = 0
_C.TRAIN.GPUS = 'GPU:0'
_C.TRAIN.learning_rate = 0.00005
_C.TRAIN.save_step = 50
_C.TRAIN.focal_loss = False
_C.TRAIN.class_balance = None


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()