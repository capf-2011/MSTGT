class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/gw/research/Tracking/pycharm_project/MSTGT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/gw/research/Tracking/pycharm_project/MSTGT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_models = '/home/gw/research/Tracking/pycharm_project/MSTGT/pretrained_models'
        self.lasot_dir = '/home/gw/research/SOT/lasot'
        self.got10k_dir = '/home/gw/research/SOT/got10k/train'
        self.got10k_val_dir = '/home/gw/research/SOT/got10k/val'
        self.lasot_lmdb_dir = '/home/gw/research/SOT/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/gw/research/SOT/got10k_lmdb'
        self.trackingnet_dir = '/home/gw/research/SOT/trackingnet'
        self.trackingnet_lmdb_dir = '/home/gw/research/SOT/trackingnet_lmdb'
        self.coco_dir = '/home/gw/research/SOT/coco'
        self.coco_lmdb_dir = '/home/gw/research/SOT/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/gw/research/SOT/vid'
        self.imagenet_lmdb_dir = '/home/gw/research/SOT/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
