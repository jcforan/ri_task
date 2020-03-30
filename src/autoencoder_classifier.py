import torch
import collections
import copy

class AutoEncoder_Based_Classifier(torch.nn.Module):
    def __init__(self, autoencoder_model, dense_layer_info, batch_norm = False):
        super(AutoEncoder_Based_Classifier, self).__init__()
        self.encoder_conv = copy.deepcopy(autoencoder_model.encoder_conv)
        self.encoder_linear = copy.deepcopy(autoencoder_model.encoder_linear)
        self.pre_linear_shape = autoencoder_model.pre_linear_shape
        def freeze(l):
            l.requires_grad = False
        for l in self.encoder_conv:
            freeze(l)
        for l in self.encoder_linear:
            freeze(l)

        hidden_layers = collections.OrderedDict()
        for i, (hs, pdropout) in enumerate(dense_layer_info):
            if i == 0:
                hidden_layers['dec_linear_0'] = torch.nn.Linear(autoencoder_model.hidden_size, hs)
            else:
                hidden_layers['dec_linear_{}'.format(i)] = torch.nn.Linear(dense_layer_info[i-1][0], hs)
            if batch_norm:
                hidden_layers['dec_batch_norm_{}'.format(i)] = torch.nn.BatchNorm1d(hs)
            
            hidden_layers['dec_relu_{}'.format(i)] = torch.nn.ReLU()
            
            if pdropout >= 0.0:
                hidden_layers['dec_dropout_{}'.format(i)] = torch.nn.Dropout(pdropout)
        
        if len(dense_layer_info) == 0:
            final_input_size = autoencoder_model.hidden_size
        else:
            final_input_size = dense_layer_info[-1][0] 
        hidden_layers['final_linear'] = torch.nn.Linear(final_input_size, 10)
        #hidden_layers['softmax'] = torch.nn.Softmax()
        self.hidden_layers = torch.nn.Sequential(hidden_layers)


    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(self.pre_linear_shape)
        x = self.encoder_linear(x)
        x = self.hidden_layers(x)
        return x
