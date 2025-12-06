from config import get_config
import torch
from network import SeqGraphAF

from engine import test


def main():
    task = 'cc'
    class_num = {
        'mf':489,
        'bp':1943,
        'cc':320
    }
    model = SeqGraphAF(num_classes=class_num[task])
    config = get_config()
    test(model, config, task)



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICE']='1'
    torch.backends.cudnn.enabled = False
    main()