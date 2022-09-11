import argparse
import torch
from utils.inference_model import model_fn

"""
This script the pytorch model into onnx format.
Execution example:
python pt2onnx.py --model_dir model

Example how run the model with ONNX Runtime:
import onnxruntime
from transformers import AutoTokenizer
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_input = tokenizer.encode_plus(
        "This is a sample",
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        )
ort_session = onnxruntime.InferenceSession("torch-model.onnx")
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input['input_ids']),
              ort_session.get_inputs()[1].name: to_numpy(model_input['attention_mask'])}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
"""

def main(args):
    model, tokenizer = model_fn(args.model_dir)
    # create a sample input
    dummy_model_input = tokenizer.encode_plus(
        "This is a sample text",
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        )
    # export to onnx
    torch.onnx.export(
        model, 
        tuple(dummy_model_input.values()),
        f="torch-model.onnx",  
        input_names=['input_ids', 'attention_mask'], 
        output_names=['logits'], 
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                    'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                    'logits': {0: 'batch_size', 1: 'sequence'}}, 
        do_constant_folding=True, 
        opset_version=13, 
    )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model', help="the directory where the model is saved as pt file")

    args = parser.parse_args()
    main(args)