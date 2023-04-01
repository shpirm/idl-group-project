import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.optim as optim
import torchvision
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import ViTForImageClassification

from sklearn.metrics import f1_score, precision_recall_fscore_support

IMG_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = len(os.listdir('./annotations'))
N_EPOCHS = 50
BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 50
LR = 0.003

class MultilabelBalancedRandomSampler(Sampler):
    """
    From https://github.com/issamemari/pytorch-multilabel-balanced-sampler

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
    
    def __init__(
            self,
            image_dir: str,
            label_dir: str,
            train_transform = None,
            test_transform = None,
            train_idx = None,
            val_idx = None
            ) -> None:
        
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
        
        return (self.train_transform(image), label) if index in self.train_idx else (self.test_transform(image), label)

def get_class_weights(dataset: ImageDataset):
    "Returns the class weights of the given multi-label dataset."
    weights = np.array([0 for labels in dataset.labels])
    for image_key in dataset.idx_to_label:
        weights += np.array(dataset.idx_to_label[image_key])
    return torch.tensor([(len(dataset) - value)/value for value in weights])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    #transforms.RandomResizedCrop(128),
    transforms.Resize((224, 224)),
    #transforms.Resize((512, 512)),
    #transforms.Pad(3),
    #transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Resize((512, 512)),
    #transforms.CenterCrop((224, 224)),
    #transforms.Pad(3),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_loader(batch_size, val_size):
    
    indices = list(range(20000))
    np.random.shuffle(indices)

    split_idx = int(np.floor(val_size * 20000))
    train_idx, val_idx = indices[split_idx:], indices[:split_idx]
    
    dataset = ImageDataset(
        './images',
        './annotations',
        train_transform,
        test_transform,
        train_idx,
        val_idx
        )

    train_sampler = MultilabelBalancedRandomSampler(
        dataset.y,
        train_idx,
        class_choice="random"
    )
    
    #train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,)
    
    return train_loader, valid_loader

train_loader, val_loader = get_loader(BATCH_SIZE_TRAIN, 0.2)

#model = models.resnet152(weights='DEFAULT')
#model = models.densenet161(weights='DEFAULT')
# model = models.vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')

# for param in model.parameters():
#     param.requires_grad = False

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

#fc layers for ViT
fc_heads = nn.Sequential(
        nn.Linear(1280, 640),
        nn.ReLU(),
        nn.BatchNorm1d(640),
        nn.Dropout(0.5),
        nn.Linear(640, 320),
        nn.ReLU(),
        nn.BatchNorm1d(320),
        nn.Dropout(0.5),
        nn.Linear(320, 160),
        nn.ReLU(),
        nn.BatchNorm1d(160),
        nn.Dropout(0.5),
        nn.Linear(160, 14)
        )

#fc layers for ResNet152
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

#fc layers for DenseNet161
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

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding = 1), # input = (3, 128, 128)
        nn.Conv2d(8, 16, 3, padding = 1),
        nn.MaxPool2d(2),
        nn.ReLU() # output = (16, 64, 64)
    )
    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(16, 32, 5, padding = 2), # input = (16, 64, 64)
        nn.Conv2d(32, 32, 3, padding = 1),
        nn.MaxPool2d(4),
        nn.ReLU() # output = (32, 16, 16)
    )
    self.conv_layer_3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding = 1), # input = (32, 16, 16)
        nn.Conv2d(64, 64, 5, padding = 2),
        nn.MaxPool2d(2),
        nn.ReLU() # output = (64, 8, 8)
    )
    self.lin_layer_1 = nn.Linear(64 * 8 * 8, 16 * 4 * 4)
    self.lin_layer_2 = nn.Linear(16 * 4 * 4, 14)

  def forward(self, x):
    x = self.conv_layer_1(x)
    x = self.conv_layer_2(x)
    x = self.conv_layer_3(x)

    x = x.view(x.size(0), -1)
    x = self.lin_layer_1(x)
    x = self.lin_layer_2(x)

    return x

#model = CNN()

#model.classifier = fc_classifier
#model.fc = fc_layers
model.classifier = fc_heads

#model.fc = nn.Linear(2048, 14)
model = model.cuda()

weights = get_class_weights(train_loader.dataset).cuda()
loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weights)

#loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

lr_sch = optim.lr_scheduler.LinearLR(optimizer, total_iters=10)
#lr_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.003)

def train(
        n_epochs,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer
        ):

    for epoch in range(n_epochs):
        model.train()
        train_correct = 0
        valid_correct = 0
        train_loss = 0.0
        valid_loss = 0.0
        epoch_mac_f1, epoch_mac_recall, epoch_mac_prec = 0.0, 0.0, 0.0
        epoch_mic_f1, epoch_mic_recall, epoch_mic_prec = 0.0, 0.0, 0.0
        epoch_v_mac_f1, epoch_v_mac_recall, epoch_v_mac_prec = 0.0, 0.0, 0.0
        epoch_v_mic_f1, epoch_v_mic_recall, epoch_v_mic_prec = 0.0, 0.0, 0.0
        
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
            
            mac_prec, mac_recall, mac_f1, mac_support = precision_recall_fscore_support(
                    target.cpu().numpy(),
                    pred.cpu().numpy(),
                    average='macro',
                    zero_division=0)
            
            mic_prec, mic_recall, mic_f1, mic_support = precision_recall_fscore_support(
                    target.cpu().numpy(),
                    pred.cpu().numpy(),
                    average='micro',
                    zero_division=0)

            epoch_mac_f1 += mac_f1*data.size(0)
            epoch_mac_recall += mac_recall*data.size(0)
            epoch_mac_prec += mac_prec*data.size(0)
            epoch_mic_f1 += mic_f1*data.size(0)
            epoch_mic_recall += mic_recall*data.size(0)
            epoch_mic_prec += mic_prec*data.size(0)

        train_loss = train_loss/len(train_loader.sampler)
        epoch_mic_f1 = epoch_mic_f1/len(train_loader.sampler)
        epoch_mic_recall = epoch_mic_recall/len(train_loader.sampler)
        epoch_mic_prec = epoch_mic_prec/len(train_loader.sampler)
        epoch_mac_f1 = epoch_mac_f1/len(train_loader.sampler)
        epoch_mac_recall = epoch_mac_recall/len(train_loader.sampler)
        epoch_mac_prec = epoch_mac_prec/len(train_loader.sampler)

        print(f'Epoch: {epoch+1}/{n_epochs}',
        f'Training Loss: {train_loss:.6f}',
        f'Train acc: {train_correct.float()/(0.8*14*len(train_loader.dataset)):.6f}',
        f'Train f1 (micro): {epoch_mic_f1:.6f}',
        f'Train rec (micro): {epoch_mic_recall:.6f}',
        f'Train prec (micro): {epoch_mic_prec:.6f}',
        f'Train f1 (macro): {epoch_mac_f1:.6f}',
        f'Train rec (macro): {epoch_mac_recall:.6f}',
        f'Train prec (macro): {epoch_mac_prec:.6f}')

        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                batch_loss = criterion(output, target)
                valid_loss += batch_loss.item()*data.size(0)
                pred = torch.sigmoid(output.data) > 0.48
                valid_correct += pred.eq(target.data.view_as(pred)).sum()
                
                v_mac_prec, v_mac_recall, v_mac_f1, v_mac_support = precision_recall_fscore_support(
                        target.cpu().numpy(),
                        pred.cpu().numpy(),
                        average='macro',
                        zero_division=0
                        )
                
                v_mic_prec, v_mic_recall, v_mic_f1, v_mic_support = precision_recall_fscore_support(
                        target.cpu().numpy(),
                        pred.cpu().numpy(),
                        average='micro',
                        zero_division=0
                        )
                
                epoch_v_mac_f1 += v_mac_f1*data.size(0)
                epoch_v_mac_recall += v_mac_recall*data.size(0)
                epoch_v_mac_prec += v_mac_prec*data.size(0)
                epoch_v_mic_f1 += v_mic_f1*data.size(0)
                epoch_v_mic_recall += v_mic_recall*data.size(0)
                epoch_v_mic_prec += v_mic_prec*data.size(0)

        valid_loss = valid_loss/len(valid_loader.sampler)
        epoch_v_mic_f1 = epoch_v_mic_f1/len(valid_loader.sampler)
        epoch_v_mic_recall = epoch_v_mic_recall/len(valid_loader.sampler)
        epoch_v_mic_prec = epoch_v_mic_prec/len(valid_loader.sampler)
        epoch_v_mac_f1 = epoch_v_mac_f1/len(valid_loader.sampler)
        epoch_v_mac_recall = epoch_v_mac_recall/len(valid_loader.sampler)
        epoch_v_mac_prec = epoch_v_mac_prec/len(valid_loader.sampler)
        
        print(f'Epoch: {epoch+1}/{n_epochs}',
        f'Validation Loss: {valid_loss:.6f}',
        f'Validation acc: {valid_correct.float()/(0.2*14*len(valid_loader.dataset)):.6f}',
        f'Validation f1 (micro): {epoch_v_mic_f1:.6f}',
        f'Validation rec (micro): {epoch_v_mic_recall:.6f}',
        f'Validation prec (micro): {epoch_v_mic_prec:.6f}',
        f'Validation f1 (macro): {epoch_v_mac_f1:.6f}',
        f'Validation rec (macro): {epoch_v_mac_recall:.6f}',
        f'Validation prec (macro): {epoch_v_mac_prec:.6f}')
        
        if (epoch == 9):
            save_model(model, epoch+1)

        model.train()
    
    return train_loss, valid_loss

def save_model(model, epoch):
    torch.save(model.state_dict(), f'modell_{epoch}.pt')

train_loss, valid_loss = train(
    10,
    model,
    train_loader,
    val_loader,
    loss,
    optimizer
    )

#test

