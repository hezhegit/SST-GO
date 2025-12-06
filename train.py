import random

import numpy as np
import torch

from config import get_config
from engine import train

from network import SeqGraphAF

def fix_random_seed(seed):
    # os.environ['CUDA_VISIBLE_DEVICE']='1'
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main():
    task = 'cc'
    class_num = {
        'mf':489,
        'bp':1943,
        'cc':320
    }
    model = SeqGraphAF(num_classes=class_num[task])
    config = get_config()
    train(model, config, task)



if __name__ == '__main__':
    fix_random_seed(216)
    main()