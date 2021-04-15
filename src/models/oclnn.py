import numpy as np
from torch import nn
from torch.nn import init
from src.util import np_to_var


class OCLNN(nn.Module):
    def __init__(self,in_chans,
                 n_classes,
                 input_time_length,
                 drop_prob=0.25,):
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.drop_prob = drop_prob

        super(OCLNN, self).__init__()

        n_filters = 16
        filter_spat_length = self.in_chans
        filter_time_length = int(self.input_time_length / 15)
        filter_time_stride = filter_time_length

        self.main_block = nn.Sequential()
        self.main_block.add_module('conv_1', nn.Conv2d(1,
                                                  n_filters,
                                                  (filter_spat_length, filter_time_length),
                                                  stride=(1, filter_time_stride),
                                                  bias=True))
        self.main_block.add_module('relu_1', nn.ReLU())
        self.main_block.add_module('drop_1', nn.Dropout(p=self.drop_prob))

        out = self.main_block(np_to_var((np.ones(
            (1, 1, self.in_chans, self.input_time_length),
            dtype=np.float32))))
        num_input_units_fc_1 = self.num_flat_features(out)

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc_1',
                                   nn.Linear(num_input_units_fc_1, self.n_classes, bias=True))
        self.classifier.add_module('logsoftmax', nn.LogSoftmax(dim=1))

        initialize_module(self.main_block)
        initialize_module(self.classifier)

    def forward(self, x):
        main_block_output = self.main_block(x)
        classifier_input = main_block_output.view(-1,
                                                   self.num_flat_features(main_block_output))
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