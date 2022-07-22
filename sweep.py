import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar 
from torchmetrics import AUROC

from utils.data import get_fold, PreprintsDataset, PreprintsDataModule
from utils.trainer_model import ModelTagger, PreprintsTagger


# Sweep parameters
hyperparameter_defaults = dict(
    learning_rate=3e-4,
    weight_decay = 1e-5,
    p_dropout=0.3
)

wandb.init(config=hyperparameter_defaults, project='preprint-tagger')
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


def main(config):
    max_token_len = 512
    n_epochs = 4
    batch_size = 32
    warmup_percentage = 0.15
    fold = 0 
    label_names = ['ml', 'cs', 'ph', 'mth', 'bio', 'fin']
    # ------------------------
    # 1 DATA PIPELINES
    # ------------------------
    df_test = pd.read_csv( Path('data/test_set.csv') )
    df_folds = pd.read_csv(Path( 'data/train_5folds.csv' ))
    df_train, df_valid = get_fold(df=df_folds, fold=fold)
    steps_per_epoch=len(df_train) // batch_size 
    total_training_steps = steps_per_epoch * n_epochs
    warmup_steps = int(total_training_steps * warmup_percentage)

    # get the tokenizer and the datamodule
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dm = PreprintsDataModule(df_train, df_valid, df_test, tokenizer, 
                        batch_size=batch_size, max_token_len=max_token_len) 
                       
    # ------------------------
    # 2 LIGHTNING MODEL
    # ------------------------
    # get the model
    model = PreprintsTagger( 
        net=ModelTagger(p_dropout=config.p_dropout, label_names=label_names),
        lr=config.learning_rate,
        wd=config.weight_decay,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        )

    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------
    wandb_logger = WandbLogger()

    # optional: log model topology
    # wandb_logger.watch(model.net)

    # ------------------------
    # 4 TRAINER
    # ------------------------
    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        max_epochs=n_epochs,
        gpus= -1 if torch.cuda.is_available() else 0,
        enable_progress_bar=True, 
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        precision=16,
        # accumulate_grad_batches=2,
        # log_every_n_steps=50,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model, dm)


if __name__ == '__main__':
    print(f'Starting a run with {config}')
    main(config)