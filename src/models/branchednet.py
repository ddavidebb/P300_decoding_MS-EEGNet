import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from src.util import np_to_var


class BranchedNet(nn.Module):
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 drop_prob=0.5):
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.drop_prob = drop_prob
        super(BranchedNet, self).__init__()

        self.main_block = nn.Sequential()
        self.main_block.add_module('conv_1', nn.Conv2d(
            1, 12, (8, 1), stride=1, bias=False,
            padding=(0, 0)))
        self.main_block.add_module('bnorm_1', nn.BatchNorm2d(12, momentum=0.01, affine=True, eps=1e-3))
        self.main_block.add_module('tanh_1', nn.Tanh())
        self.main_block.add_module('drop_1', nn.Dropout(p=self.drop_prob))

        self.tail_block1 = nn.Sequential()
        self.tail_block1.add_module('conv_1', nn.Conv2d(
            12, 4, (1, 12), stride=1, bias=False,
            padding=(0, 0)))
        self.tail_block1.add_module('bnorm_1', nn.BatchNorm2d(4, momentum=0.01, affine=True, eps=1e-3))
        self.tail_block1.add_module('tanh_1', nn.Tanh())
        self.tail_block1.add_module('drop_1', nn.Dropout(p=self.drop_prob))
        self.tail_block1.add_module('pool_1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.tail_block1.add_module('conv_2', nn.Conv2d(
            4, 8, (1, 12), stride=1, bias=False,
            padding=(0, 0)))
        self.tail_block1.add_module('bnorm_2', nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3))
        self.tail_block1.add_module('tanh_2', nn.Tanh())
        self.tail_block1.add_module('drop_2', nn.Dropout(p=self.drop_prob))

        self.tail_block1.add_module('conv_3',nn.Conv2d(
            8, 8, (1, 12), stride=1, bias=False,
            padding=(0, 0)))
        self.tail_block1.add_module('bnorm_3', nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3))
        self.tail_block1.add_module('tanh_3', nn.Tanh())
        self.tail_block1.add_module('drop_3', nn.Dropout(p=self.drop_prob))

        self.tail_block2 = nn.Sequential()
        self.tail_block2.add_module('conv_1', nn.Conv2d(
            12, 4, (1, 52), stride=1, bias=False,
            padding=(0, 0)))
        self.tail_block2.add_module('bnorm_1', nn.BatchNorm2d(4, momentum=0.01, affine=True, eps=1e-3))
        self.tail_block2.add_module('tanh_1', nn.Tanh())
        self.tail_block2.add_module('drop_1', nn.Dropout(p=self.drop_prob))
        self.tail_block2.add_module('pool_1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))

        out = self.main_block(np_to_var(np.ones(
            (1, 1, self.in_chans, self.input_time_length),
            dtype=np.float32)))

        num_input_units_fc_1 = self.num_flat_features(self.tail_block1(out)) + \
                               self.num_flat_features(self.tail_block2(out))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc_1',
                                   nn.Linear(num_input_units_fc_1, self.n_classes, bias=True))
        self.classifier.add_module('logsoftmax', nn.LogSoftmax(dim=1))

        initialize_module(self.main_block)
        initialize_module(self.tail_block1)
        initialize_module(self.tail_block2)
        initialize_module(self.classifier)

    def forward(self, x):
        main_block_output = self.main_block(x)
        tail_block1_output = self.tail_block1(main_block_output)
        tail_block2_output = self.tail_block2(main_block_output)
        tail_block1_output = tail_block1_output.view(-1, self.num_flat_features(tail_block1_output))
        tail_block2_output = tail_block2_output.view(-1, self.num_flat_features(tail_block2_output))
        classifier_input = th.cat((tail_block1_output, tail_block2_output), 1)
        classifier_output = self.classifier(classifier_input)
        return classifier_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def initialize_module(module):
    for mod in module.modules():
        if hasattr(mod, 'weight'):
            if not ('BatchNorm' in mod.__class__.__name__):
                init.xavier_uniform_(mod.weight, gain=1)
            else:
                init.constant_(mod.weight, 1)
        if hasattr(mod, 'bias'):
            if mod.bias is not None:
                init.constant_(mod.bias, 0)