import os
import PIL
import torch
import pickle
import random
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

torch.manual_seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True, help='Path to the images')
parser.add_argument('-d', '--data-split', dest='data', required=True, help='Dataset name')
parser.add_argument('-f', '--features', dest='features', required=True, help='Path to the features file')
parser.add_argument('-m', '--model', dest='model', required=True,
                    help='swav/path_to_the_pre-trained_model.')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                    required=False, default=100, help='Batch size.')
args = parser.parse_args()
dataset = args.data

with open(f'../data_splits/{dataset}-split.pkl', 'rb') as f:
    data_partition = pickle.load(f)

class RSDataset(Dataset):
  def __init__(self, image_dir, image_list, transform=None):
    self.image_dir = image_dir
    self.image_list = image_list
    self.transform = transform

  def __len__(self):
    return(len(self.image_list['filename']))

  def __getitem__(self, idx):
    path = os.path.join(self.image_dir, self.image_list['filename'][idx])
    img = PIL.Image.open(path)
    if self.transform:
      img = self.transform(img)
      
    label = self.image_list['label'][idx]

    return img, label

class Rot90:
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, [1, 2])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('../models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.train()
        print("num parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def validate(val_loader, model):
    """
    evaluation
    """
    features = []
    targets = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            print(idx)
            input = input.float()
            targets.append(target.numpy())
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            feat = model(input)

            features.append(feat.view(feat.size(0), -1).cpu().numpy())

    return features, targets

image_datasets = {x: RSDataset(args.path, data_partition[x], data_transforms[x]) 
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model == 'swav':
    model = ResNetBackbone(args.model)
else:
    model = torch.load(args.model)
    model = model.module[0]

for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

features_train, targets_train = validate(dataloaders['train'], model)
features_train = np.concatenate(features_train, axis = 0)
targets_train = np.concatenate(targets_train, axis = 0)
print(features_train.shape)
print(targets_train.shape)

features_test, targets_test = validate(dataloaders['test'], model)
features_test = np.concatenate(features_test, axis = 0)
targets_test = np.concatenate(targets_test, axis = 0)
print(features_test.shape)
print(targets_test.shape)

np.savez(args.features, x_train=features_train, y_train=targets_train,
         x_test=features_test, y_test=targets_test) 
