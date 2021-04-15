import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from src.util import np_to_var


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                     maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                     maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


class BranchedEEGNet(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 F1=8,
                 D=2,
                 F2=16,
                 first_pool_size=4,
                 first_pool_stride=4,
                 temporal_kernel_length=64+1,
                 separable_kernel_lengths=[16+1],
                 separable_num_convs=[],
                 separable_pooling_length=0,
                 drop_prob=0.5,
                 max_norm_conv_spatial=1.,
                 max_norm_fc=0.25):
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.fist_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.temporal_kernel_length = temporal_kernel_length
        self.separable_kernel_lengths = separable_kernel_lengths
        self.separable_pooling_length = separable_pooling_length
        self.drop_prob = drop_prob
        self.max_norm_conv_spatial=max_norm_conv_spatial
        self.max_norm_fc = max_norm_fc

        super(BranchedEEGNet, self).__init__()

        self.main_block = nn.Sequential()
        self.main_block.add_module('conv_1', nn.Conv2d(
            1, self.F1, (1, self.temporal_kernel_length), stride=1, bias=False,
            padding=(0, self.temporal_kernel_length // 2,)))
        self.main_block.add_module('bnorm_1', nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3))

        if self.max_norm_conv_spatial is None:
            self.main_block.add_module('conv_2',
                                        nn.Conv2d(self.F1, self.F1 * self.D, (self.in_chans, 1),
                                                  stride=1, bias=False, groups=self.F1, padding=(0, 0))
                                       )
        else:
            self.main_block.add_module('conv_2',
                                       Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.in_chans, 1),
                                                            stride=1, bias=False, groups=self.F1, padding=(0, 0),
                                                            max_norm=self.max_norm_conv_spatial)
                                       )
        self.main_block.add_module('bnorm_2', nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3))
        self.main_block.add_module('elu_1', nn.ELU())
        self.main_block.add_module('avg_pool_1',
                                   nn.AvgPool2d(
                                       kernel_size=(1, self.fist_pool_size), stride=(1, self.first_pool_stride)))
        self.main_block.add_module('drop_1', nn.Dropout(p=self.drop_prob))

        out = self.main_block(np_to_var(np.ones(
            (1, 1, self.in_chans, self.input_time_length),
            dtype=np.float32)))

        tails = []
        num_input_units_fc_1 = 0
        for i, separable_kernel_length in enumerate(self.separable_kernel_lengths):
            tail = nn.Sequential()
            for j in range(separable_num_convs[i]):
                tail.add_module('conv_{0}'.format(j), nn.Conv2d(
                    self.F1 * self.D, self.F1 * self.D, (1, separable_kernel_length),
                    stride=1, bias=False, groups=self.F1 * self.D, padding=(0, (separable_kernel_length // 2))))
            tail.add_module('conv_{0}'.format(j+1), nn.Conv2d(
                self.F1 * self.D, self.F2, (1, 1),
                stride=1, bias=False, padding=(0, 0)))
            tail.add_module('bnorm_1', nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3))
            tail.add_module('elu_1', nn.ELU())
            tail.add_module('avg_pool_1',nn.AvgPool2d(
                kernel_size=(1, 2 * self.fist_pool_size),
                stride=(1, 2 * self.first_pool_stride)))
            tail.add_module('drop_1', nn.Dropout(p=self.drop_prob))
            tails.append(tail)

            num_input_units_fc_1 += self.num_flat_features(tail(out))

        if self.separable_pooling_length != 0:
            pool_tail = nn.Sequential()
            if self.separable_pooling_length != 1:
                pool_tail.add_module("avg_pool_1", nn.AvgPool2d(kernel_size=(1, self.separable_pooling_length),
                                                                stride=(1, 1),
                                                                padding=(0, self.separable_pooling_length//2)))
            pool_tail.add_module('conv_1', nn.Conv2d(self.F1 * self.D, self.F2,
                                                     (1, 1), stride=1, bias=False, padding=(0, 0)))
            pool_tail.add_module('bnorm_1', nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3))
            pool_tail.add_module('elu_1', nn.ELU())
            pool_tail.add_module('avg_pool_2', nn.AvgPool2d(kernel_size=(1, 2 * self.fist_pool_size),
                                                            stride=(1, 2 * self.first_pool_stride)))
            pool_tail.add_module('drop_1', nn.Dropout(p=self.drop_prob))
            tails.append(pool_tail)

            num_input_units_fc_1 += self.num_flat_features(pool_tail(out))

        self.tails = nn.ModuleList(tails)


        self.classifier = nn.Sequential()
        if self.max_norm_fc is None:
            self.classifier.add_module('fc_1',
                                       nn.Linear(num_input_units_fc_1, self.n_classes, bias=True)
                                       )
        else:
            self.classifier.add_module('fc_1',
                                       LinearWithConstraint(num_input_units_fc_1, self.n_classes, bias=True, max_norm=self.max_norm_fc)
                                       )

        self.classifier.add_module('logsoftmax', nn.LogSoftmax(dim=1))

        initialize_module(self.main_block)
        for t in self.tails:
            initialize_module(t)
        initialize_module(self.classifier)

    def forward(self, x):
        main_block_output = self.main_block(x)
        tail_outputs = []
        for t in self.tails:
            tail_output = t(main_block_output)
            tail_outputs.append(tail_output.view(-1, self.num_flat_features(tail_output)))
        classifier_input = th.cat(tail_outputs, 1)
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