#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: lightning_mepave_datamodule.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging

import pytorch_lightning as pl
from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.datasets import MEPAVEDataset, Split

logger = logging.getLogger(__name__)


class MEPAVEDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MEPAVEDataset(
                self.cfg.data_dir,
                self.tokenizer,
                self.transform,
                self.cfg.max_seq_length,
                self.cfg.vocab_path,
                Split.train)
            self.val_dataset = MEPAVEDataset(
                self.cfg.data_dir,
                self.tokenizer,
                self.transform,
                self.cfg.max_seq_length,
                self.cfg.vocab_path,
                Split.valid)
        if stage == "test" or stage is None:
            self.test_dataset = MEPAVEDataset(
                self.cfg.data_dir,
                self.tokenizer,
                self.transform,
                self.cfg.max_seq_length,
                self.cfg.vocab_path,
                Split.test)

    def prepare_data(self):
        logger.info(f"please place dataset in {self.cfg.data_dir}.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.val_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.val_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)
