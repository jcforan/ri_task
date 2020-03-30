import torch
import numpy as np
import copy

from .autoencoder import AutoEncoder
from .classifier import Classifier
from .autoencoder_classifier import AutoEncoder_Based_Classifier
from .criterion import autoencoder_loss_fn, classifier_loss_fn
from .util import get_class_labels, get_cross_entropy_weights

def train(model,
          lossfn,
          optimizer,
          train_data,
          valid_data,
          bs,
          early_stopping_criteria,
          weight = None,
          calc_accuracy=False,
          noise_std=None,
):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=bs)

    using_gpu = torch.cuda.is_available()
    if using_gpu:
        model = model.cuda()
    epoch_no = 0
    best_score = None
    best_params = None
    losses = []
    while True:
        train_loss = 0
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            if using_gpu:
                images = images.cuda()
                labels = labels.cuda()
                if weight is not None:
                    weight = weight.cuda()
            optimizer.zero_grad()
            orig_images = images
            if noise_std is not None:
                noise = torch.empty(images.shape).normal_(mean=0, std = noise_std)
                if using_gpu:
                    noise = noise.cuda()
                images = images + noise
            output = model(images)
            loss = lossfn(output, orig_images, labels, weight)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= step # Average loss per image
        valid_loss = 0
        model.eval()
        correct = 0
        for step, (images, labels) in enumerate(valid_loader):
            if using_gpu:
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                output = model(images)
                loss = lossfn(output, images, labels)
            valid_loss += loss
            if calc_accuracy:
                _, pred = torch.max(output, 1)
                correct += (pred == labels).sum().item()
        valid_loss /= step

        if calc_accuracy:
            accuracy = correct / len(valid_data)
            score  = -accuracy # negate to maximise
        else:
            score = valid_loss

        if (best_score is None) or score < best_score:
            best_score = score
            best_params = copy.deepcopy(model.state_dict())

        losses.append((train_loss, valid_loss))
        if calc_accuracy:
            print("Epoch: {}, train loss: {}, validation loss: {} accuracy: {:.2f}".format(
                epoch_no, train_loss, valid_loss, accuracy*100.0))
        else:
            print("Epoch: {}, train loss: {}, validation loss: {}".format(epoch_no, train_loss, valid_loss))
        if early_stopping_criteria(epoch_no, train_loss, valid_loss):
            return best_params, losses
        epoch_no += 1

def stopping_criteria_fn_builder(max_updates = None, max_epochs = None):
    best_epoch=None
    best_valid_loss = None
    assert (max_updates is not None) or (max_epochs is not None), "Either max_updates or max_epochs must be specified"
    def stopping_criteria_fn(epoch, train_loss, valid_loss):
        nonlocal best_epoch, best_valid_loss
        if (best_valid_loss is None) or (valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            best_epoch = epoch
        if max_epochs is not None and epoch >= max_epochs:
            return True
        if ((max_updates is not None) and
            ((epoch - best_epoch) >= max_updates)):
            return True
        return False
    return stopping_criteria_fn

def train_autoencoder(conv_layer_info,
                      dense_layer_info,
                      train_dataset,
                      valid_dataset,
                      batch_norm,
                      batch_size,
                      max_epochs=500,
                      valid_stop_count=None,
                      lr = 1e-3,
                      noise_std = None):

    autoencoder = AutoEncoder(conv_layer_info,
                              dense_layer_info,
                              batch_norm=batch_norm)
    print(autoencoder)
    stopping_criteria_fn = stopping_criteria_fn_builder(valid_stop_count, max_epochs)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    best_autoencoder_params, losses = train(autoencoder,
                                            autoencoder_loss_fn,
                                            optimizer,
                                            train_dataset,
                                            valid_dataset,
                                            batch_size,
                                            stopping_criteria_fn,
                                            noise_std=noise_std
    )

    # Restore best parameters to model
    autoencoder.load_state_dict(best_autoencoder_params)

    return autoencoder, losses

def train_autoencoder_classifier(autoencoder,
                                 classifier_hidden_sizes,
                                 train_dataset,
                                 valid_dataset,
                                 batch_norm,
                                 batch_size,
                                 max_epochs=500,
                                 valid_stop_count=None,
                                 lr = 1e-3):

    classifier = AutoEncoder_Based_Classifier(autoencoder, classifier_hidden_sizes, batch_norm)
    print(classifier)
    stopping_criteria_fn = stopping_criteria_fn_builder(valid_stop_count, max_epochs)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    best_classifier_params, losses = train(classifier,
                                           classifier_loss_fn,
                                           optimizer,
                                           train_dataset,
                                           valid_dataset,
                                           batch_size,
                                           stopping_criteria_fn,
                                           get_cross_entropy_weights(),
                                           calc_accuracy = True)
    classifier.load_state_dict(best_classifier_params)
    return classifier, losses

def train_classifier(conv_layer_details,
                     dense_layer_details,
                     train_dataset,
                     valid_dataset,
                     batch_norm,
                     batch_size,
                     max_epochs=500,
                     valid_stop_count = None,
                     lr = 1e-3):

    classifier = Classifier(conv_layer_details, dense_layer_details, batch_norm)
    print(classifier)
    stopping_criteria_fn = stopping_criteria_fn_builder(valid_stop_count, max_epochs)

    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=lr)
    best_classifier_params, losses = train(classifier,
                                           classifier_loss_fn,
                                           optimizer,
                                           train_dataset,
                                           valid_dataset,
                                           batch_size,
                                           stopping_criteria_fn,
                                           get_cross_entropy_weights(),
                                           calc_accuracy = True)
    classifier.load_state_dict(best_classifier_params)
    return classifier, losses

def test_autoencoder(autoencoder, test_data, batch_size):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    total_loss = 0
    using_gpu = torch.cuda.is_available()
    if using_gpu:
        autoencoder = autoencoder.cuda()

    autoencoder.eval()
    n = [0] * 10
    for i, (images, labels) in enumerate(test_loader):
        if using_gpu:
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            output = autoencoder(images)
            loss = autoencoder_loss_fn(output, images, labels)
        total_loss += loss
    print("Total Mean Loss: {:.2f}".format(total_loss/i))


def test_classifier(classifier, test_dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    total_test_loss = 0
    class_labels = get_class_labels()
    no_classes = len(class_labels)
    class_match_count = [0] * no_classes
    class_count = [0] * no_classes
    for step, (images, labels) in enumerate(test_loader):
        classifier.eval()
        #images = images.view(-1, 3*32*32)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        output = classifier(images)
        _, pred = torch.max(output, 1)
        for l, p in zip(labels, pred):
            c = int(l)
            if p == l:
                class_match_count[c] += 1
            class_count[c] += 1

        loss = classifier_loss_fn(output, images, labels)
        total_test_loss += loss.item()
    print("Mean Test Loss: {}".format(total_test_loss/step))
    accuracy = sum(class_match_count)/sum(class_count)
    for c in range(no_classes):
        print("Class: {}: Match: {:.2f}%".format(class_labels[c],
                                        class_match_count[c]*100.0/class_count[c]))
    print("Total accuracy: {:.2f}%".format(accuracy*100.0))
    return accuracy

def train_autoencoder_and_classifier(auto_encoder_conv_layers,
                                     auto_encoder_hidden_sizes,
                                     classifier_hidden_sizes,
                                     datasets,
                                     batch_norm,
                                     batch_size):
    train_data, valid_data, test_data = datasets
    print("=====================\n\nTraining Autoencoder\n\n=====================\n")
    autoencoder, _ = train_autoencoder(auto_encoder_conv_layers,
                                                        auto_encoder_hidden_sizes,
                                                        train_data,
                                                        valid_data,
                                                        batch_norm,
                                                        batch_size,
                                                        valid_stop_count=3)
    print("=====================\n\nTesting Autoencoder\n\n=====================\n")
    test_autoencoder(autoencoder, test_data)
    print("=====================\n\nTraining Classifier\n\n=====================\n")
    classifier, _ = train_autoencoder_classifier(autoencoder,
                                                 classifier_hidden_sizes,
                                                 train_data,
                                                 valid_data,
                                                 True,
                                                 batch_size,
                                                 valid_stop_count=3)
    print("=====================\n\nTesting Classifier\n\n=====================\n")
    accuracy = test_classifier(classifier,
                               test_data)
    return accuracy, autoencoder, classifier
