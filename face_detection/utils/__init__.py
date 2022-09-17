from .augmentations import SSDAugmentation
from .transforms import BaseTransform
import torch.nn as nn
import torch.nn.init as init


def adjust_learning_rate(lr, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# ''' FIX THIS '''
#         if iteration * (epoch + 1) in cfg['lr_steps']:
#             step_index += 1
#             adjust_learning_rate(optimizer, args.gamma, step_index)
