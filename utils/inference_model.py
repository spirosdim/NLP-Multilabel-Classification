from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel 
import pytorch_lightning as pl


# --------------------------------------------------------------------------------------------
# ------------------------------------------ Inference nn.Module -----------------------------
# --------------------------------------------------------------------------------------------
 

class Tagger(nn.Module):
    """
    Minimal model class just for inference
    """
    def __init__(self):
        super().__init__()
        self.bert_model_name='distilbert-base-uncased'
        self.label_names=['ml', 'cs', 'ph', 'mth', 'bio', 'fin']
        config = AutoConfig.from_pretrained(self.bert_model_name)
        self.bert =  AutoModel.from_config(config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(self.label_names))
        self.sigmoid_fnc = torch.nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.last_hidden_state[:,0]) #taking the ouput from [CLS] token 
        output = self.sigmoid_fnc(output)    
        return output


# --------------------------------------------------------------------------------------------
# ------------------------------------------ Inference functions -----------------------------
# --------------------------------------------------------------------------------------------

def model_fn(model_dir):
    model = Tagger()
    with open(Path(model_dir) / 'model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f))
    tokenizer = AutoTokenizer.from_pretrained(model.bert_model_name)
    return model, tokenizer 
    

def predict_fn(abstract, model, tokenizer):   
    label_nms = ['Machine Learning', 'Computer Science', 'Physics', 'Mathematics', 'Biology', 'Finance-Economics']
    # Tokenize abstract
    encoded_input = tokenizer.encode_plus(
        abstract,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        )
    # Compute token embeddings
    model.eval()
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    get_result = {}
    for i in range(len(label_nms)):       
        get_result[label_nms[i]] = round(model_output.tolist()[0][i], 3)

    # return dictonary
    return get_result