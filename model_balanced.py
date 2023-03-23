import numpy as np
import pandas as pd
import os
import random

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
from sklearn.metrics import classification_report, precision_recall_fscore_support

from transformers import ViTForImageClassification

from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = '../%s'

IMG_SIZE = 224
N_EPOCHS = 10
LR = 0.0001
BATCH_SIZE_TRAIN = 75
BATCH_SIZE_TEST = 50
VAL_SIZE = 0.2

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
      """
      Returns the names of the labels and a dictionary
      where each image index corresponds to a list of its labels. 
      """

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
    def __init__(self, image_dir: str, label_dir: str, train_transform = None, test_transform = None, train_idx = None) -> None:
        self.paths = [os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)]

        self.train_transform = train_transform
        self.test_transform = test_transform

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
        
        return (self.train_transform(image), label) if index in self.train_idx else (self.test_transform(image), label)

def get_class_weights(dataset: ImageDataset):
    "Returns the class weights of the given multi-label dataset."
    weights = np.array([0 for labels in dataset.labels])
    for image_key in dataset.idx_to_label:
        weights += np.array(dataset.idx_to_label[image_key])

    return torch.tensor([(len(dataset) - value)/value for value in weights])


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def get_loader(data: ImageDataset, batch_size, val_size):

    data_len = len(data)
    indices = list(range(data_len))

    np.random.shuffle(indices)
    split_idx = int(val_size * data_len)

    train_idx, val_idx = indices[split_idx:], indices[:split_idx]

    train_sampler = MultilabelBalancedRandomSampler(dataset.y, train_idx, class_choice="random")
    data.train_idx = train_idx

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    return train_loader, valid_loader

dataset = ImageDataset(DATA_DIR % 'images', DATA_DIR % 'annotations', train_transform, test_transform)
train_loader, valid_loader = get_loader(dataset, BATCH_SIZE_TRAIN, VAL_SIZE)

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = nn.Linear(in_features=768, out_features=len(dataset.labels), bias=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)
weights = get_class_weights(dataset)

loss = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_sch = optim.lr_scheduler.LinearLR(optimizer, total_iters=10)

def train(n_epochs, model, train_loader, valid_loader, criterion, optimizer):
    for epoch in range(n_epochs):
        train_correct = 0
        valid_correct = 0
        train_loss, train_f1, train_rec, train_prec = 0.0, 0.0, 0.0, 0.0
        valid_loss, valid_f1, valid_rec, valid_prec = 0.0, 0.0, 0.0, 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).logits

            loss = criterion(output, target)
            # loss = (loss * weights.to(device))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_sch.step()

            pred = torch.sigmoid(output.data) > 0.48
            train_correct += pred.eq(target.data.view_as(pred)).sum()
            prec, recall, f1, support = precision_recall_fscore_support(
                target.cpu().numpy(), pred.cpu().numpy(), average='samples', zero_division=1)
            
            train_loss += loss * data.size(0)
            train_f1   += f1 * data.size(0)
            train_rec  += recall * data.size(0)
            train_prec += prec * data.size(0)
        
        train_loss /= len(train_loader.sampler)
        train_f1 /= len(train_loader.sampler)
        train_rec /= len(train_loader.sampler)
        train_prec /= len(train_loader.sampler)

        train_correct = train_correct.float() / (len(dataset.labels) * len(train_loader.sampler))
        print(f'Epoch: {epoch+1} / {n_epochs}',
        f'Training Loss: {train_loss:.6f}',
        f'Train acc: {train_correct:.6f}',
        f'Train f1: {train_f1:.6f}',
        f'Train rec: {train_rec:.6f}',
        f'Train prec: {train_prec:.6f}')

        # pred = np.array(pred.cpu(), dtype=float)
        # print(classification_report(
        #       target.cpu(),
        #       pred,
        #       output_dict=False,
        #       target_names=dataset.labels
        #       ))
        
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).logits
                loss = criterion(output, target).mean()

                pred = torch.sigmoid(output.data) > 0.48
                valid_correct += pred.eq(target.data.view_as(pred)).sum()
                prec, recall, f1, support = precision_recall_fscore_support(
                    target.cpu().numpy(), pred.cpu().numpy(), average='samples', zero_division=1)
                
                valid_loss += loss * data.size(0)
                valid_f1   += f1 * data.size(0)
                valid_rec  += recall * data.size(0)
                valid_prec += prec * data.size(0)
            
            valid_loss /= len(valid_loader.sampler)
            valid_f1 /= len(valid_loader.sampler)
            valid_rec /= len(valid_loader.sampler)
            valid_prec /= len(valid_loader.sampler)

            valid_correct = valid_correct.float() / (len(dataset.labels) * len(valid_loader.sampler))
            print(f'Epoch: {epoch+1} / {n_epochs}',
            f'Validation Loss: {valid_loss:.6f}',
            f'Validation acc: {valid_correct:.6f}',
            f'Validation f1: {valid_f1:.6f}',
            f'Validation rec: {valid_rec:.6f}',
            f'Validation prec: {valid_prec:.6f}')

            # pred = np.array(pred.cpu(), dtype=float)
            # print(classification_report(
            #       target.cpu(),
            #       pred,
            #       output_dict=False,
            #       target_names=dataset.labels
            #       ))
                
        model.train()

    return train_loss, valid_loss

train_loss, valid_loss = train(N_EPOCHS, model, train_loader, valid_loader, loss, optimizer)

# should be tested

# def test(model, test_loader):
#     test_loss = 0.0
#     test_f1 = 0.0
#     correct = 0
#     model.eval()
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.cuda(), target.cuda()
#             output = model.forward(data)
#             batch_loss = loss(output, target)
#             test_loss += batch_loss.item()*data.size(0)
#             pred = torch.sigmoid(output.data) > 0.48
#             correct += pred.eq(target.data.view_as(pred)).sum()
#             test_f1 += f1_score(target.cpu().numpy(),
#                                 pred.cpu().numpy(),
#                                 average='samples',
#                                 zero_division=1
#                                 )*data.size(0)
    
#     test_loss = test_loss/len(test_loader.sampler)
#     test_f1 = test_f1/len(test_loader.sampler)
    
#     print(f'Test Loss: {test_loss:.6f}... ',
#     f'Test Accuracy: {correct.float()/(14*len(test_loader.sampler)):.6f}',
#     f'Test f1: {test_f1:.6f}')

# #test(model, test_loader) 
