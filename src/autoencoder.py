import torch
import collections

class AutoEncoder(torch.nn.Module):
    def __init__(self, conv_layer_info, dense_layer_info, batch_norm=False):
        super(AutoEncoder, self).__init__()
        encoder_conv_layers = collections.OrderedDict()

        sizes = []

        prev_channels = 3
        cur_side = 32

        # Encoder convolutional layers
        for i, (channels, pool_size, pdropout) in enumerate(conv_layer_info):
            encoder_conv_layers['conv_{}'.format(i)] = torch.nn.Conv2d(prev_channels, channels, 3,padding=1)
            encoder_conv_layers['relu_{}'.format(i)] = torch.nn.ReLU()
            if batch_norm:
                encoder_conv_layers['batch_norm_{}'.format(i)] = torch.nn.BatchNorm2d(channels)
            if pool_size > 1:
                encoder_conv_layers['maxpool_{}'.format(i)] = torch.nn.MaxPool2d(pool_size)
                cur_side = cur_side // pool_size
            if pdropout >= 0.0:
                encoder_conv_layers['dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)
            prev_channels = channels

        self.encoder_conv = torch.nn.Sequential(encoder_conv_layers)

        # Encoder linear layers
        self.pre_linear_shape = (-1, prev_channels*cur_side*cur_side)
        encoder_linear_layers = collections.OrderedDict()
        prev_size = prev_channels * cur_side * cur_side
        for i, (hs, pdropout) in enumerate(dense_layer_info):
            encoder_linear_layers['linear_{}'.format(i)] = torch.nn.Linear(prev_size, hs)

            if batch_norm:
                encoder_linear_layers['batch_norm_{}'.format(i)] = torch.nn.BatchNorm1d(hs)

            encoder_linear_layers['relu_{}'.format(i)] = torch.nn.ReLU()

            if pdropout >= 0.0:
                encoder_linear_layers['dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)

            prev_size = hs

        self.hidden_size = prev_size
        self.encoder_linear = torch.nn.Sequential(encoder_linear_layers)

        # Decoder linear layers
        decoder_linear_layers = collections.OrderedDict()
        rev_dense_layers = list(reversed(dense_layer_info))
        for i, (hs, pdropout) in enumerate(rev_dense_layers):
            next_size = rev_dense_layers[i+1][0] if (i < len(rev_dense_layers) - 1) else self.pre_linear_shape[1]
            decoder_linear_layers['linear_{}'.format(i)] = torch.nn.Linear(hs, next_size)

            if batch_norm:
                decoder_linear_layers['batch_norm_{}'.format(i)] = torch.nn.BatchNorm1d(next_size)

            decoder_linear_layers['relu_{}'.format(i)] = torch.nn.ReLU()

            if pdropout >= 0.0:
                decoder_linear_layers['dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)


        self.decoder_linear = torch.nn.Sequential(decoder_linear_layers)
        self.post_linear_shape = (-1, prev_channels, cur_side, cur_side)

        # Decoder convolutional layers
        decoder_conv_layers = collections.OrderedDict()
        reverse_layers = list(reversed(conv_layer_info))
        for i, (channels, pool_size, pdropout) in enumerate(reverse_layers):
            is_last_layer = i == (len(reverse_layers) -1)
            next_channels = 3 if is_last_layer else reverse_layers[i+1][0]

            if pool_size > 1:
                decoder_conv_layers['dec_upscale_{}'.format(i)] = torch.nn.Upsample(scale_factor=pool_size)

            decoder_conv_layers['dec_conv_{}'.format(i)] = torch.nn.Conv2d(channels, next_channels, 3, padding=1)

            if batch_norm:
                decoder_conv_layers['batch_norm_{}'.format(i)] = torch.nn.BatchNorm2d(next_channels)

            if is_last_layer:
                decoder_conv_layers['dec_sigmoid_{}'.format(i)] = torch.nn.Sigmoid()
            else:
                decoder_conv_layers['dec_relu_{}'.format(i)] = torch.nn.ReLU()

            if pdropout >= 0.0:
                decoder_conv_layers['dec_dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)

        self.decoder_conv = torch.nn.Sequential(decoder_conv_layers)

    def forward(self, x):
        x = self.encoder_conv(x)
        x  = x.view(self.pre_linear_shape)
        x = self.encoder_linear(x)
        x = self.decoder_linear(x)
        x = x.view(self.post_linear_shape)
        x = self.decoder_conv(x)
        return x
