import os
import re
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torchvision
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import ViTForImageClassification, ViTFeatureExtractor

from sklearn.metrics import f1_score, precision_recall_fscore_support, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

IMG_SIZE = 224
NUM_CHANNELS = 3
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 250
BATCH_SIZE_TEST = 50
LR = 0.0008


def parse():
    par = ArgumentParser()

    par.add_argument(
        '--model',
        '-m',
        dest='model',
        default='vit',
        help='Select model to train: vit, resnet, densenet, cnn. Default: vit')
    
    par.add_argument(
        '--train_img_path',
        '-i',
        dest='train_dir',
        default='./train_images',
        help='Path to training image directory'
    )

    par.add_argument(
        '--train_ann_path',
        '-a',
        dest='ann_dir',
        default='./annotations',
        help='Path to training label directory'
    )

    par.add_argument(
        '--test_img_path',
        '-t',
        dest='test_dir',
        default='./test_images',
        help='Path to test image directory'
    )
    
    return par.parse_args()

args = parse()

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
    def __init__(
            self,
            image_dir: str,
            label_dir: str,
            train_transform = None,
            test_transform = None,
            train_idx = None
            ) -> None:
        
        self.paths = [os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)]
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_idx = train_idx
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

vit_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

crd_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

train_transform = vit_transform if args.model == 'vit' else crd_transform

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
#%%
dataset = ImageDataset(args.train_dir, args.ann_dir, train_transform, test_transform)
#%%
def get_loader(data: ImageDataset, batch_size, val_size):
    """ 
    Returns the training and validation loader of the given multi-label dataset.
    Uses MultilabelBalancedRandomSampler to balanced the data.
    """

    data_len = len(data)
    indices = list(range(data_len))

    np.random.shuffle(indices)
    split_idx = int(val_size * data_len)

    train_idx, val_idx = indices[split_idx:], indices[:split_idx]

    train_sampler = MultilabelBalancedRandomSampler(dataset.y, train_idx, class_choice="random")
    data.train_idx = train_idx

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    return train_loader, valid_loader

train_loader, val_loader = get_loader(dataset, BATCH_SIZE_TRAIN, 0.2)


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

def get_model():
    if args.model == 'cnn':
        model = CNN()
        return model
    
    if args.model == 'vit':
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        for param in model.parameters():
            param.requires_grad = False

        fc_heads = nn.Sequential(
            nn.Linear(768, 640),
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
        
        model.classifier = fc_heads

    elif args.model == 'resnet':
        model = models.resnet152(weights='DEFAULT')
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
        
        model.fc = fc_layers
            
    elif args.model == 'densenet':
        model = models.densenet161(weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False

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
    
        model.classifier = fc_classifier

    return model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = get_model()

model = model.to(device)
weights = get_class_weights(train_loader.dataset).to(device)

loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

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
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if args.model == 'vit':
                output = model(data).logits
            else:
                output = model(data)
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
        f'Train acc: {train_correct.float()/(len(dataset.labels)*len(train_loader.sampler)):.6f}',
        f'Train f1 (micro): {epoch_mic_f1:.6f}',
        f'Train rec (micro): {epoch_mic_recall:.6f}',
        f'Train prec (micro): {epoch_mic_prec:.6f}',
        f'Train f1 (macro): {epoch_mac_f1:.6f}',
        f'Train rec (macro): {epoch_mac_recall:.6f}',
        f'Train prec (macro): {epoch_mac_prec:.6f}')

        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)

                if args.model == 'vit':
                    output = model(data).logits
                else:
                    output = model(data)
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
        f'Validation acc: {valid_correct.float()/(len(dataset.labels)*len(valid_loader.sampler)):.6f}',
        f'Validation f1 (micro): {epoch_v_mic_f1:.6f}',
        f'Validation rec (micro): {epoch_v_mic_recall:.6f}',
        f'Validation prec (micro): {epoch_v_mic_prec:.6f}',
        f'Validation f1 (macro): {epoch_v_mac_f1:.6f}',
        f'Validation rec (macro): {epoch_v_mac_recall:.6f}',
        f'Validation prec (macro): {epoch_v_mac_prec:.6f}')
        
        #if (epoch == 9):
        #    save_model(model, epoch+1)

        model.train()
    
    return train_loss, valid_loss

def save_model(model, epoch):
    torch.save(model.state_dict(), f'model_{epoch}.pt')

train_loss, valid_loss = train(N_EPOCHS, model, train_loader, val_loader, loss, optimizer)


# To generate test answers
class TestDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            transform = None,
            ) -> None:
        
        self.paths = [os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir)]
        self.transform = transform
        
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')
 
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int):
        "Returns one sample of data."
        image = self.load_image(index)
        image_name = os.path.splitext(os.path.basename(self.paths[index]))[0]

        return (self.transform(image), image_name) if self.transform else (image, image_name)
    
test_set = TestDataset(
    args.test_dir,
    test_transform
    )

test_loader = DataLoader(test_set, BATCH_SIZE_TEST)

#model.load_state_dict(torch.load('./asd.pt'))

preds, names_list = [], []
model.eval()
with torch.no_grad():
    for data, names in test_loader:
        data = data.to(device)
        output = model(data).logits
        pred = torch.sigmoid(output.data) > 0.48
        preds.append(pred.cpu().numpy())
        names_list.append(names)

preds = np.array(preds)
names_list = np.array(names_list)
preds = preds.reshape(-1, preds.shape[-1])
names_list = names_list.reshape(-1, names_list.shape[-1])
names_list = names_list.flatten()

df1 = pd.DataFrame(preds)
df1.insert(0,'image',names_list)
df1 = df1.drop(columns=['Unnamed: 0'])
df1 = df1*1
df1['image'].replace(
    {r"^.{1,2}" : ''},
    inplace=True,
    regex=True)
df1 = df1.sort_values(by='image')
df1 = df1.rename(columns={
    'image': 'Filename',
    '0': 'portrait',
    '1': 'dog',
    '2': 'clouds',
    '3': 'male',
    '4': 'baby',
    '5': 'people',
    '6': 'tree',
    '7': 'female',
    '8': 'sea',
    '9': 'car',
    '10': 'night',
    '11': 'river',
    '12': 'flower',
    '13': 'bird'
})

df1 = pd.concat([
    df1[['Filename']],
    df1[df1.columns.difference(['Filename'])
        ].sort_index(axis=1)],
        ignore_index=False,
        axis=1)
df1 = df1.set_index('Filename')
df1 = df1.reset_index(drop=False)

df1.to_csv('fin_preds.csv')

with open("./fin_preds.csv", 'r', encoding='utf-8') as csvin, open("./test_set_preds.tsv", 'w', newline='', encoding='utf-8') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')
 
    for row in csvin:
        tsvout.writerow(row)

# To generate plots for error analysis
y_pred, y = [], []
mis_dict = {name : [] for name in range(len(dataset.labels))}
with torch.no_grad():
    model.eval()
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch).logits
        pred = (torch.sigmoid(y_test_pred.data) > 0.48).long()
        
        pred = pred.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        
        mis = y_batch != pred
        for i in range(len(mis)):
          for j in range(len(mis[0])):
            if mis[i, j] == 1:
              mis_dict[j].append((X_batch[i].cpu().numpy(),
                                  y_batch[i][j]))
        
        for i in pred: y_pred.append(i)
        for i in y_batch: y.append(i)

y = np.array(y)
y_pred = np.array(y_pred)

m = multilabel_confusion_matrix(y, y_pred)

f, axes = plt.subplots(2, 7, figsize=(30, 10))
axes = axes.ravel()

for i in range(14):
    disp = ConfusionMatrixDisplay(m[i])
    disp.plot(ax=axes[i], values_format='')
    disp.ax_.set_title(f'class {dataset.labels[i]}')
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.2, hspace=0.1)
plt.show()

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225]),
    transforms.Normalize(mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.]),
    ])

test_labels = ['male', 'female', 'people', 'portrait']
for label in test_labels:
    f, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes = axes.ravel()

    index = dataset.labels.index(label)
    examples = [random.randint(0, len(mis_dict[index])) for i in range(4)]
    for i in range(4):
        im = torch.from_numpy(mis_dict[index][examples[i]][0])
        axes[i].imshow(invTrans(im).permute(1, 2, 0))

        cl = mis_dict[index][examples[i]][1]
        axes[i].set_title('False Positive' if cl == 0 else 'False Negative')
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.2)
    f.suptitle(f"Class {label}")
    plt.show()
