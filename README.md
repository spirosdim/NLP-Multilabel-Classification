# NLP-Multilabel-Classification
The scope of this project is to apply a number of tools on solving a deep learning task. An NLP Multilabel classification problem is chosen.

## To DO:

- [X] Create the dataset from arxiv (arxiv api, pandas, numpy, sklearn)
- [X] Visualize the data (pandas, seaborn, matplotlib)
- [X] fine-tune a BERT family model (transformers, pytorch, pytorch-lightning, hydra) 
- [X] hyperparameter search (wandb sweep)
- [X] inference script 
- [ ] Docker file for inference
- [ ] convert to onnx 
- [ ] testing (pytest)



## Hyperparameter tuning using Sweep from WandB
On the terminal: 
    1. `wandb login `
    2. `wandb sweep sweep.yaml`
    3. `wandb agent <name_of_sweep_shown_on_step2> --count 10`
