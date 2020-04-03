import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.parse_cfg import parse_config, parse_data_config

def make_modules(block_defs):
    hyperparams = block_defs.pop(0)
    out_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for module_m, module_def in enumerate(block_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            batch_norm = int(module_def['batch_normalize'])
            filters = int(module_def['filter'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1 // 2)
            modules.add_module(
                f'conv_{module_m}',
                nn.Conv2d(
                    in_channels=out_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias=not batch_norm
                ),
            )
            if batch_norm:
                modules.add_module(
                    f'batch_norm__{module_m}',nn.BatchNorm2d(filters=filters, momentum=0.9, eps=1e-5))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_relu_{module_m}', nn.LeakyReLU(negative_slope=0.1))

        elif module_def['type']

