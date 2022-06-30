import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer 

import pytorch_lightning as pl


def get_fold(df, fold):
    """
    get one fold of train and validation sets
    Args: 
        df (pandas DataFrame): a dataframe with information for the papers and a kfold column with the fold number
        fold (int): the number of the fold to extract
    Returns:
        df_train (pandas DataFrame): a dataframe with the training observations and targets
        df_valid (pandas DataFrame): a dataframe with the validation observations and targets
        
    Example:
        df_train, df_valid = get_fold(df=df_folds, fold=0)
    """
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    df_train.drop("kfold", axis=1, inplace=True)
    df_valid.drop("kfold", axis=1, inplace=True)
    return df_train, df_valid
    
    
    
    
# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Dataset ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

class PreprintsDataset:

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    self.label_columns = self.data.columns.tolist()[1:]
    abstract_text = data_row.text

    labels = data_row[self.label_columns]

    encoding = self.tokenizer.encode_plus(
      abstract_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )
    
    

# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------- DataModule ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------

class PreprintsDataModule(pl.LightningDataModule):

  def __init__(self, train_df, valid_df, test_df, tokenizer, batch_size=8, max_token_len=128, n_workers=4):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.valid_df = valid_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.n_workers = n_workers

  def setup(self, stage=None):
    self.train_dataset = PreprintsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.valid_dataset = PreprintsDataset(
      self.valid_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = PreprintsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.n_workers
    )

  def val_dataloader(self):
    return DataLoader(
      self.valid_dataset,
      batch_size=self.batch_size,
      num_workers=self.n_workers
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=self.n_workers
    )
    
    
