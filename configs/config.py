from dataclasses import dataclass

@dataclass
class General:
    cwdir: str

@dataclass
class Dataset:
    test: str
    folds: str
    fold: int
    label_names: list[str]

@dataclass
class Train:
    pre_model_name: str
    lr: float
    warmup_percentage: float
    batch_size: int
    n_epochs: int
    use_gpu: bool
    model_dir: str
    show_bar: bool
    max_token_len: int

@dataclass
class Logs:
    log_project: str
    log_pr_name: str

@dataclass
class TypesConfig:
    general: General
    dataset: Dataset
    train: Train
    logs: Logs