import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cpu", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_baseline"

model_path = "image_enhacement/weights/SRResNet_x4-ImageNet-2096ee7f.pth.tar"