import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import random
from .util import unpickle, cifar_data_path, get_label_no, get_class_labels
from PIL import Image

class ReducedCifarDataset(torch.utils.data.Dataset):
    def __init__(self, files, class_labels, reduce_labels, reduction_rate, add_transforms=None):
        batch_dicts = []
        for f in files:
            batch_dicts.append(unpickle(f))
        all_data = {}
        for i, bd in enumerate(batch_dicts):
            if i == 0:
                 for k in bd:
                    if k == b'batch_label':
                        continue
                    all_data[k] = bd[k]
            else:
                for k in bd:
                    if k == b'batch_label':
                        continue
                    if isinstance(bd[k], list):
                        all_data[k].extend(bd[k])
                    elif isinstance(bd[k], np.ndarray):
                        all_data[k] = np.concatenate((all_data[k], bd[k]))
        for i,l in enumerate(class_labels):
            (idxs, ) = np.where(np.array(all_data[b'labels']) == get_label_no(l))
            if l in reduce_labels:
                np.random.shuffle(idxs)
                idxs = idxs[:int(len(idxs)* reduction_rate)]
            if i == 0:
                use_idxs = idxs[:]
            else:
                use_idxs = np.concatenate((use_idxs, idxs))
        use_idxs = np.sort(use_idxs)
        self.labels = np.array(all_data[b'labels'])[use_idxs]
        self.data = np.transpose(all_data[b'data'][use_idxs, :].reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        if add_transforms is  None:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = add_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_data = self.data[idx, :, :, :]
        #Convert to PIL image to keep consistent with other datasets
        image_data = Image.fromarray(image_data)
        image_data = self.transforms(image_data)

        return (image_data, label)

def get_datasets(reduced_train_labels,
                 reduction = 0.5,
                 do_random_hflip = False):

    if do_random_hflip:
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    else:
        train_transform = None

    class_labels = get_class_labels()
    train_data_files = glob.glob(os.path.join(cifar_data_path, "data_batch_*"))
    test_data_files = glob.glob(os.path.join(cifar_data_path, "test_batch"))
    train_dataset = ReducedCifarDataset(train_data_files, class_labels, reduced_train_labels, reduction, train_transform)
    orig_test_dataset = ReducedCifarDataset(test_data_files, class_labels, [], 1.0, None)


    n = len(orig_test_dataset)
    valid_n = n // 2
    idx = list(range(n))
    random.shuffle(idx)
    valid_idx = idx[:valid_n]
    test_idx = idx[valid_n:]
    valid_dataset = torch.utils.data.Subset(orig_test_dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(orig_test_dataset, test_idx)

    print("{} training samples, {} validation samples, {} test samples".format(len(train_dataset),
                                                                               len(valid_dataset),
                                                                               len(test_dataset)))
    return train_dataset, valid_dataset, test_dataset

