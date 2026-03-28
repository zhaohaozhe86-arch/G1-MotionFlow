
import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.utils.feature_to_joints import recover_g1_motion
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .utils import humanml3d_collate


class UnitreeG1DataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'unitreeg1'
        self.name = "unitreeg1"
        self.njoints = 30 
        
        # Path to the dataset
        data_root = cfg.DATASET.UNITREEG1.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs') 
        
        # Mean and std of the dataset
        dis_data_root = cfg.DATASET.UNITREEG1.MEAN_STD_PATH
        self.hparams.mean = np.load(pjoin(dis_data_root, "Mean.npy")) 
        self.hparams.std = np.load(pjoin(dis_data_root, "Std.npy"))

        self.hparams.mean_eval = self.hparams.mean
        self.hparams.std_eval = self.hparams.std
        
        # Length of the dataset 
        self.hparams.max_motion_length = cfg.DATASET.UNITREEG1.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.UNITREEG1.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.UNITREEG1.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.UNITREEG1.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                self.hparams.win_size = 64
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.UNITREEG1.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        print(f"Dataset Initialized. Detected Feature Dimension: {self.nfeats}")
        
        
    def feats2joints(self, features):

        # global_positions: (B, T, J, 3) 
        # global_rotations: (B, T, J, 4)
        global_positions, global_rotations = recover_g1_motion(
            features, self.hparams.mean, self.hparams.std
        )
        
        should_save = self.hparams.debug
        if hasattr(self, 'trainer') and self.trainer is not None:
             if getattr(self.trainer, 'testing', False) or getattr(self.trainer, 'predicting', False):
                 should_save = True

        if should_save:
            pos_np = global_positions.detach().cpu().numpy()   # (B, T, J, 3)
            rot_np = global_rotations.detach().cpu().numpy()   # (B, T, J, 4)
            
            save_path = os.path.join(self.hparams.data_root, 'inference_raw_g1.npz')
            
            np.savez(
                save_path,
                global_positions=pos_np,
                global_rotations=rot_np,
                meta={'coordinate': 'y-up', 'quat_order': 'wxyz', 'desc': 'output from MotionGPT'}
            )

        return global_positions.view(features.shape[0], features.shape[1], -1) 
    
    def joints2feats(self, features):
        raise NotImplementedError("G1 joints2feats not implemented yet. Use pre-processed features.")
    
    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features
    
    def renorm4t2m(self, features):
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list