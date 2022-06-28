import torch
import torch.nn as nn

from transformers import AutoModel, get_linear_schedule_with_warmup 

import pytorch_lightning as pl
from torchmetrics import AUROC

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------- Training LightningModule ----------------------------------------
# ----------------------------------------------------------------------------------------------------------

class PreprintsTagger(pl.LightningModule):
    
  def __init__(self, lr, bert_model_name=None, n_training_steps=None, n_warmup_steps=None, label_names=None):
    super().__init__()
    if label_names is None:
        label_names=['ml', 'cs', 'ph', 'mth', 'bio', 'fin']
    if bert_model_name is None:
        bert_model_name='distilbert-base-uncased'

    self.label_names = label_names
    self.lr = lr
    self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, len(label_names))
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCEWithLogitsLoss()
    self.sigmoid_fnc = torch.nn.Sigmoid()
    for lbl in label_names: globals()[f"self.auroc_{lbl}"] = AUROC()

    
    self.save_hyperparameters()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.last_hidden_state[:,0]) #taking the ouput from [CLS] token    
    return output
    
    

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
      self.logger.experiment.add_scalar(f"{name}_auroc/Train", class_roc_auc, self.current_epoch)
      
      
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
      self.logger.experiment.add_scalar(f"{name}_auroc/Valid", class_roc_auc, self.current_epoch)


  def configure_optimizers(self):

    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  

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




