# NLP-Multilabel-Classification
The scope of this side-project is to apply a number of tools on solving a deep learning task. An NLP Multilabel classification problem is chosen.

## To DO:

- [X] Create the dataset from arxiv (arxiv api, pandas, numpy, sklearn)
- [X] Visualize the data (pandas, seaborn, matplotlib)
- [X] fine-tune a BERT family model (transformers, pytorch, pytorch-lightning, hydra) 
- [X] hyperparameter search (wandb sweep)
- [X] inference script 
- [X] Docker file for inference
- [X] convert to onnx 

## Create an environment
The project dependencies can be installed as follows (using [miniconda](https://docs.conda.io/en/latest/miniconda.html)):

* `conda create --name nlp_abstracts python=3.9 -y`
* `conda activate nlp_abstracts`
* `pip install -r requirements.txt`
* `python -m ipykernel install --user --name=nlp_abstracts`

## Data Collection and Visualization
To see how the dataset is downloaded and some visualization to inspect it, see `DataCollectVisual.ipynb`.


## Fine-tune a BERT family model for multi-target classification
Used libraries: `pytorch`, `pytorch-lightning`, `transformers` for the model, `hydra` to create the config.yaml file and the define the dataclasses in `configs/config.py`.

See `finetune.py`


## Hyperparameter tuning using Sweep from WandB
Hyperparameter tuning using bayesian optimization. Need a wandb account (I highly recomment wandb for monitoring your model runs).

To try it out, on the terminal: 

* `wandb login`
* `wandb sweep sweep.yaml`
* `wandb agent <name_of_sweep_displayed_from_step2> --count 10`

## Inference script
See `inference_pt.py` for inference with the model saved in .pt file.


## Dockerfile
Created a first approach to a Dockerfile in case of need to dockerize a procedure.

## Onnx file for inference
Create an onnx file to run inference with onnxruntime.
See `pt2onnx.py`.

## Inference in AWS SageMaker
On how to create a real-time endpoint in AWS SageMaker, you can see my post on medium [Inference your own NLP trained model on aws sagemaker with pytorchmodel or huggingefacemodel](https://medium.com/innovation-res/inference-your-own-nlp-trained-model-on-aws-sagemaker-with-pytorchmodel-or-huggingefacemodel-30bcbdc4348b)