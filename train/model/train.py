import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    current_dir = Path(__file__).parent
    yaml_path = "./fisheye8k.yaml"
    check_path(yaml_path)
    model = RTDETR('./SO_DETR/ultralytics/cfg/models/A-Test-r50-M.yaml')