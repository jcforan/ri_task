import torch
import collections

class Classifier(torch.nn.Module):
    def __init__(self, conv_layer_info, dense_layer_info, batch_norm = False):
        super(Classifier, self).__init__()
        conv_layers = collections.OrderedDict()

        prev_channels = 3
        cur_side = 32

        for i, (channels, pool_size, pdropout) in enumerate(conv_layer_info):
            conv_layers['conv_{}'.format(i)] = torch.nn.Conv2d(prev_channels, channels, 3,padding=1)
            if batch_norm:
                conv_layers['conv_batch_norm_{}'.format(i)] = torch.nn.BatchNorm2d(channels)
            conv_layers['conv_relu_{}'.format(i)] = torch.nn.ReLU()

            if pool_size > 1:
                conv_layers['conv_maxpool_{}'.format(i)] = torch.nn.MaxPool2d(pool_size)
                cur_side = cur_side // pool_size

            if pdropout >= 0.0:
                conv_layers['conv_dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)

            prev_channels = channels

        self.conv_layers = torch.nn.Sequential(conv_layers)
        self.pre_linear_shape = prev_channels * cur_side * cur_side

        hidden_layers = collections.OrderedDict()

        for i, (hs, pdropout) in enumerate(dense_layer_info):
            if i == 0:
                hidden_layers['lin_linear_0'] = torch.nn.Linear(self.pre_linear_shape, hs)
            else:
                hidden_layers['lin_linear_{}'.format(i)] = torch.nn.Linear(dense_layer_info[i-1][0], hs)
            if batch_norm:
                hidden_layers['lin_batch_norm_{}'.format(i)] = torch.nn.BatchNorm1d(hs)

            hidden_layers['lin_relu_{}'.format(i)] = torch.nn.ReLU()

            if pdropout > 0.0:
                hidden_layers['lin_dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)

        if len(dense_layer_info) == 0:
            final_input_size = self.pre_linear_shape
        else:
            final_input_size = dense_layer_info[-1][0]
        hidden_layers['final_linear'] = torch.nn.Linear(final_input_size, 10)
        self.hidden_layers = torch.nn.Sequential(hidden_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.pre_linear_shape)
        x = self.hidden_layers(x)
        return x
