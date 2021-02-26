#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: run_mjave_bert.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging
import os
import datetime

from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers import WandbLogger

from src.lightning_classes import MEPAVEDataModule
from src.lightning_classes import MjaveBert


logger = logging.getLogger(__name__)


def run(cfg):
    pl.seed_everything(cfg.training.seed)

    output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, "lightning_logs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    # logger
    wandb_logger = WandbLogger(
        name="mjave_bert_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        save_dir=output_dir,
        project="mepave"
    )

    # initialize trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        **cfg.trainer
    )

    # datamodule & sampler
    dm = MEPAVEDataModule(cfg=cfg.datamodule)
    dm.setup()

    # model
    model = MjaveBert(
        hparams=cfg.lightningmodule
    )

    # train model
    trainer.fit(model=model, datamodule=dm)

    # test model
    trainer.test(model=model, datamodule=dm)


@hydra.main(config_path="conf", config_name="config_mjave_bert")
def run_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    run_model()
