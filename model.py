import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.parse_cfg import parse_config, parse_data_config

def make_module_list(block_defs):
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
                    f'batch_norm__{module_m}',nn.BatchNorm2d(num_features=filters, momentum=0.9, eps=1e-5))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_relu_{module_m}', nn.LeakyReLU(negative_slope=0.1))

        elif module_def["type"] == 'upsample':
            upsample = Upsample(scale=int(module_def['stride']), mode='nearest')
            modules.add_module(f'upsample_{module_m}', upsample)

        elif module_def["type"] == "route":
            layers = [int(l) for l in module_def['layers'].split(',')]
            filters = sum([out_filters[1:][l] for l in layers])
            modules.add_module(f'route_{module_m}', EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = out_filters[1:][int(module_def['from'])]
            modules.add_module(f'shortcut_{module_m}', EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(m) for m in module_def['mask'].split(',')]
            anchors = [int(a) for a in module_def['anchors'].split(',')]
            anchors = [(anchors[a], anchors[a + 1]) for a in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_size = int(hyperparams['height'])
            detect_layer = YoloLayer(anchors, num_classes, img_size)
            modules.add_module(f'yolo_{module_m}', detect_layer)
        module_list.append(modules)
        out_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    def __init__(self, scale, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale = scale
        self.mode = mode


class YoloLayer(nn.module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.anchor_count = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0
        self.thresh = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        gs = self.grid_size
        self.stride = self.img_dim / gs
        self.grid_x = torch.arange(gs).repeat(gs, 1).view([1, 1, gs, gs]).type(torch.FloatTensor)
        self.grid_y = torch.arange(gs).repeat(gs, 1).t().view([1, 1, gs, gs]).type(torch.FloatTensor)
        self.scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.anchor_count, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.anchor_count, 1, 1))

    def forward(self, x, y=None, img_dim=None):
        self.img_dim = img_dim
        samples = x.size(0)
        grid_size = x.size(2)
        prediction = (
            x.view(samples, self.anchor_count, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        x = torch.sigmoid((prediction[...., 0]))
        y = torch.sigmoid(prediction[...., 1])
        w = prediction[...., 2]
        h = prediction[...., 3]
        pred_conf = torch.sigmoid(prediction[...., 4])
        pred_class = torch.sigmoid(prediction[...., 5])
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)
