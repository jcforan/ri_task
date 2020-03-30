import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import time
import os

global class_labels
class_labels = None
reduced_train_labels = ['bird', 'deer', 'truck']
script_dir=os.path.realpath(os.path.dirname(__file__))
cifar_data_path = os.path.realpath(os.path.join(script_dir, "..", "data", "cifar-10-batches-py"))

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_for_imshow(image):
    image = image/2 + 0.5
    npimg = image.numpy()
    return np.transpose(npimg, (1, 2, 0))

def imshow(img):
    plt.imshow(convert_for_imshow(img))
    plt.show()

def show_grid_images(h, w, images, save_filename):
    assert len(images) == h*w
    def make_row(l, r):
        return np.hstack(convert_for_imshow(images[i]) for i in range(l, r))
    grid = np.vstack((make_row(i*w, (i+1)*w) for i in range(h)))
    plt.imshow(grid)
    plt.savefig(save_filename)
    plt.show()

def plot_losses(losses, title, filename):
    train_losses, valid_losses = zip(*losses)
    epoch = list(range(len(losses)))
    plt.plot(epoch, train_losses, label="Training loss")
    plt.plot(epoch, valid_losses, label="Validation loss")
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss")
    plt.legend(loc=1)
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def show_sample_image(dataset):
    idx = np.random.randint(0, len(dataset))
    image, label_no = dataset[idx]
    print("Image {}: {}".format(idx, class_labels[label_no]))
    imshow(image)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def download_data():
    if not os.path.isdir(cifar_data_path):
        parent_dir=os.path.dirname(cifar_data_path)
        os.makedirs(parent_dir, exist_ok=True)
        os.system("curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o {}/cifar-10-python.tar.gz".format(parent_dir))
        os.system("cd {} && tar xf cifar-10-python.tar.gz".format(parent_dir))
        assert os.path.isdir(cifar_data_path), "Failed to get cifar-10 data"

def read_meta_data():
    global class_labels
    if class_labels is None:
        if not os.path.isdir(cifar_data_path):
            download_data()
        raw_meta_data = unpickle(os.path.join(cifar_data_path, "batches.meta"))
        meta_data = {}
        for k in raw_meta_data:

            if isinstance(raw_meta_data[k], int):
                meta_data[k.decode()] = raw_meta_data[k]
            elif isinstance(raw_meta_data[k], list):
                meta_data[k.decode()] = [l.decode() for l in raw_meta_data[k]]
        class_labels = meta_data['label_names']

def get_label_no(label):
    global class_labels
    if class_labels is None:
        read_meta_data()
    if label in class_labels:
        return class_labels.index(label)
    return -1

def get_class_labels():
    if class_labels is None:
        read_meta_data()
    return class_labels

def get_cross_entropy_weights():
    weights = np.array([2.0 if c in reduced_train_labels else 1.0 for c in class_labels])
    return torch.Tensor(weights/weights.sum()) #Normalize
