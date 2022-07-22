import torch
import torch.nn as nn

from transformers import AutoModel, get_linear_schedule_with_warmup 

import pytorch_lightning as pl
from torchmetrics import AUROC

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------- Training LightningModule ----------------------------------------
# ----------------------------------------------------------------------------------------------------------

class ModelTagger(nn.Module):
    """
    model class
    """
    def __init__(self, p_dropout=0.1, bert_model_name='distilbert-base-uncased', label_names=['ml', 'cs', 'ph', 'mth', 'bio', 'fin']):
        super().__init__()
        self.p_dropout=p_dropout
        self.bert_model_name=bert_model_name
        self.label_names=label_names
        self.bert =  AutoModel.from_pretrained(self.bert_model_name, return_dict=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(self.label_names))
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output.last_hidden_state[:,0]) #taking the ouput from [CLS] token  
        output = self.classifier(output)  
        return output

class PreprintsTagger(pl.LightningModule):
    
  def __init__(self, net, lr, wd, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.lr = lr
    self.wd = wd
    self.net = net
    self.label_names = self.net.label_names
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    for lbl in self.label_names: globals()[f"self.auroc_{lbl}"] = AUROC()
    self.save_hyperparameters(ignore=['net'])
    self.criterion = nn.BCEWithLogitsLoss()
    self.sigmoid_fnc = torch.nn.Sigmoid()
    

  def forward(self, input_ids, attention_mask, labels=None):
    return self.net(input_ids, attention_mask=attention_mask)  


  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    outputs = self(input_ids, attention_mask, labels)
    loss = 0
    if labels is not None:
        loss = self.criterion(outputs, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs.detach(), "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    outputs = self(input_ids, attention_mask, labels)
    loss = 0
    if labels is not None:
        loss = self.criterion(outputs, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs.detach(), "labels": labels}

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    outputs = self(input_ids, attention_mask, labels)
    loss = 0
    if labels is not None:
        loss = self.criterion(outputs, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss


  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i, name in enumerate(self.label_names):
      class_roc_auc = globals()[f"self.auroc_{name}"](self.sigmoid_fnc(predictions[:, i].long()), labels[:, i])
      self.log(f"{name}_auroc/Train", class_roc_auc)#, self.current_epoch)
      
      
  def validation_epoch_end(self, validation_step_outputs):
    labels = []
    predictions = []
    for output in validation_step_outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(self.label_names):
      class_roc_auc = globals()[f"self.auroc_{name}"](self.sigmoid_fnc(predictions[:, i].long()), labels[:, i])
      self.log(f"{name}_auroc/Valid", class_roc_auc) #, self.current_epoch)


  def configure_optimizers(self):

    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)  

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )