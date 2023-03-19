import numpy as np
import pandas as pd
import os
import random
import time
import json

from collections import defaultdict
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from PIL import Image
import matplotlib.pyplot as plt

NUM_CHANNELS = 3
WIDTH = HEIGHT = 128
NUM_CLASSES = len(os.listdir('./annotations'))
N_EPOCHS = 50
BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 50
LR = 0.01

# Multilabel sampler based on: https://github.com/issamemari/pytorch-multilabel-balanced-sampler
class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)

def get_labels(labels_dir, images_dir):
    label_names = [os.path.splitext(f)[0] for f in os.listdir(labels_dir)]
    image_indexes = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]

    label_indexes = {file_name: set() for file_name in label_names}
    for file_name in label_names:
        with open(os.path.join(labels_dir, file_name + '.txt')) as f:
            for line in f: label_indexes[file_name].add(line.strip())

    image_labels = {index: [] for index in image_indexes}
    for label in label_names:
        for key in image_labels.keys():
            image_labels[key].append(int(str(key[2:]) in label_indexes[label]))

    return label_names, image_labels

class ImageDataset(Dataset):
    
    def __init__(self, image_dir: str, label_dir: str, train_transform = None, test_transform = None, train_idx = None, val_idx = None) -> None:
        self.paths = [os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)]
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.labels, self.idx_to_label = get_labels(label_dir, image_dir)

        class_probabilities = np.random.random([14])
        class_probabilities = class_probabilities / sum(class_probabilities)
        class_probabilities *= 2
        self.y = (
            np.random.random([len(self.paths), 14]) < class_probabilities
        ).astype(int)


    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        image = self.load_image(index)
        image_name = os.path.splitext(os.path.basename(self.paths[index]))[0]
        label = torch.Tensor(self.idx_to_label[image_name])
        l = len(self.paths)
        
        if index in self.train_idx:
            return self.train_transform(image), label
        elif index in self.val_idx:
            return self.test_transform(image), label


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_loader(batch_size, val_size):

    indices = list(range(20000))
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_size * 20000))

    train_idx, val_idx = indices[split_idx:], indices[:split_idx]
    dataset = ImageDataset('./images', './annotations', train_transform, test_transform, train_idx, val_idx)
    train_sampler = MultilabelBalancedRandomSampler(dataset.y, train_idx, class_choice="random")
    valid_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,)

    return train_loader, valid_loader

train_loader, val_loader = get_loader(BATCH_SIZE_TRAIN, 0.2)

#model = models.resnet50(weights='DEFAULT')
#model = models.resnet34(weights='DEFAULT')
#model = models.resnet101(weights='DEFAULT')
model = models.resnet152(weights='DEFAULT')
#model = models.densenet161(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

fc_layers = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 14)
        )

fc_classifier = nn.Sequential(
        nn.Linear(2208, 1104),
        nn.ReLU(),
        nn.BatchNorm1d(1104),
        nn.Dropout(0.5),
        nn.Linear(1104, 552),
        nn.ReLU(),
        nn.BatchNorm1d(552),
        nn.Dropout(0.5),
        nn.Linear(552, 276),
        nn.ReLU(),
        nn.BatchNorm1d(276),
        nn.Dropout(0.5),
        nn.Linear(276, 14)
        )

#model.classifier = fc_classifier
model.fc = fc_layers

#model.fc = nn.Linear(2048, 14)
model = model.cuda()

loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01)
#lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.01)
lr_sch = optim.lr_scheduler.LinearLR(optimizer, total_iters=10)


def train(
        n_epochs,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer
        ):

    for epoch in range(n_epochs):
        train_correct = 0
        valid_correct = 0
        train_loss, train_f1, train_rec, train_prec = 0.0, 0.0, 0.0, 0.0
        valid_loss, valid_f1, valid_rec, valid_prec = 0.0, 0.0, 0.0, 0.0
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lr_sch.step()
            train_loss += loss.item()*data.size(0)
            pred = torch.sigmoid(output.data) > 0.48
            train_correct += pred.eq(target.data.view_as(pred)).sum()
            train_f1 += f1_score(target.cpu().numpy(),
                                 pred.cpu().numpy(),
                                 average='samples',
                                 zero_division=1
                                 )*data.size(0)
            train_rec += recall_score(target.cpu().numpy(),
                                        pred.cpu().numpy(),
                                        average='samples',
                                        zero_division=1
                                        )*data.size(0)
            train_prec += precision_score(target.cpu().numpy(),
                                        pred.cpu().numpy(),
                                        average='samples',
                                        zero_division=1
                                        )*data.size(0)
        
        train_loss = train_loss/(0.8*len(train_loader.sampler))
        train_f1 = train_f1/(0.8*len(train_loader.sampler))
        train_rec = train_rec/(0.8*len(train_loader.sampler))
        train_prec = train_prec/(0.8*len(train_loader.sampler))
        
        print(f'Epoch: {epoch+1}/{n_epochs}',
        f'Training Loss: {train_loss:.6f}',
        f'Train acc: {train_correct.float()/(14*len(train_loader.sampler)):.6f}',
        f'Train f1: {train_f1:.6f}',
        f'Train rec: {train_rec:.6f}',
        f'Train prec: {train_prec:.6f}')
        
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                batch_loss = criterion(output, target)
                valid_loss += batch_loss.item()*data.size(0)
                pred = torch.sigmoid(output.data) > 0.48
                valid_correct += pred.eq(target.data.view_as(pred)).sum()
                valid_f1 += f1_score(target.cpu().numpy(),
                                     pred.cpu().numpy(),
                                     average='samples',
                                     zero_division=1
                                     )*data.size(0)
                valid_rec += recall_score(target.cpu().numpy(),
                                            pred.cpu().numpy(),
                                            average='samples',
                                            zero_division=1
                                            )*data.size(0)
                valid_prec += precision_score(target.cpu().numpy(),
                                            pred.cpu().numpy(),
                                            average='samples',
                                            zero_division=1
                                            )*data.size(0)
                
                
        valid_loss = valid_loss/len(valid_loader.sampler)
        valid_f1 = valid_f1/len(valid_loader.sampler)
        valid_rec = valid_rec/len(valid_loader.sampler)
        valid_prec = valid_prec/len(valid_loader.sampler)
        
        print(f'Epoch: {epoch+1}/{n_epochs}',
        f'Validation Loss: {valid_loss:.6f}',
        f'Validation acc: {valid_correct.float()/(14*len(valid_loader.sampler)):.6f}',
        f'Validation f1: {valid_f1:.6f}',
        f'Validation rec: {valid_rec:.6f}',
        f'Validation prec: {valid_prec:.6f}')
        model.train()

    return train_loss, valid_loss

train_loss, valid_loss = train(
    10,
    model,
    train_loader,
    val_loader,
    loss,
    optimizer
    )

def test(model, test_loader):
    test_loss = 0.0
    test_f1 = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model.forward(data)
            batch_loss = loss(output, target)
            test_loss += batch_loss.item()*data.size(0)
            pred = torch.sigmoid(output.data) > 0.48
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_f1 += f1_score(target.cpu().numpy(),
                                pred.cpu().numpy(),
                                average='samples',
                                zero_division=1
                                )*data.size(0)
    
    test_loss = test_loss/len(test_loader.sampler)
    test_f1 = test_f1/len(test_loader.sampler)
    
    print(f'Test Loss: {test_loss:.6f}... ',
    f'Test Accuracy: {correct.float()/(14*len(test_loader.sampler)):.6f}',
    f'Test f1: {test_f1:.6f}')

#test(model, test_loader)
