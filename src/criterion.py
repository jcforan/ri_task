import torch

def autoencoder_loss_fn(output, images, labels, weight = None):
    criterion = torch.nn.MSELoss(reduction='mean')
    return criterion(output, images)

def classifier_loss_fn(output, images, labels, weight = None):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = weight)
    labels = labels.long().view(-1)

    return criterion(output, labels)
