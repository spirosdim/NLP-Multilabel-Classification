import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar 
from torchmetrics import AUROC
import hydra
from omegaconf import OmegaConf, DictConfig

from utils.data import get_fold, PreprintsDataset, PreprintsDataModule
from utils.trainer_model import PreprintsTagger




@hydra.main(config_path='configs', config_name='config')
def finetune(cfg: DictConfig):
    df_test = pd.read_csv( Path(cfg.general.cwdir) / cfg.dataset.test)
    df_folds = pd.read_csv(Path(cfg.general.cwdir) / cfg.dataset.folds)
    df_train, df_valid = get_fold(df=df_folds, fold=cfg.dataset.fold)
    steps_per_epoch=len(df_train) // cfg.train.batch_size  
    total_training_steps = steps_per_epoch * cfg.train.n_epochs
    warmup_steps = int(total_training_steps * cfg.train.warmup_percentage)
    
    model = PreprintsTagger( 
      lr=cfg.train.lr,
      bert_model_name = cfg.train.pre_model_name, 
      n_warmup_steps=warmup_steps,
      n_training_steps=total_training_steps,
      label_names=cfg.dataset.label_names
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.pre_model_name)
    dm = PreprintsDataModule(df_train, df_valid, df_test, tokenizer, 
                            batch_size=cfg.train.batch_size, max_token_len=cfg.train.max_token_len)
                         
                         
                         
    logger = TensorBoardLogger(cfg.logs.log_project, name=cfg.logs.log_pr_name)
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if cfg.train.show_bar: callbacks.append(TQDMProgressBar(refresh_rate=40))
    
    trainer = pl.Trainer(
      logger=logger,
      enable_checkpointing=True,
      callbacks=callbacks,
      max_epochs=cfg.train.n_epochs,
      auto_select_gpus=cfg.train.use_gpu,
      enable_progress_bar=cfg.train.show_bar, 
      gradient_clip_val=0.6,
      gradient_clip_algorithm="value",
      precision=16,
      log_every_n_steps=50,
    )



    trainer.fit(model, dm)
    
    
    
    mdl_dir = Path(cfg.general.cwdir) / cfg.train.model_dir
    mdl_dir.mkdir(parents=True, exist_ok=True)
    
    with open(mdl_dir / 'model.pth', 'wb') as f:
        torch.save(model.state_dict(), f)
    
    
if __name__=='__main__':
    finetune()