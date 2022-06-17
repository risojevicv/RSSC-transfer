import os
import PIL
import time
import copy
import torch
import pickle
import random
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_warmup as warmup
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(42)
cudnn.benchmark = True

EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', dest='path', required=True, help='Path to the images')
parser.add_argument('-d', '--data-split', dest='data', required=True, help='Dataset name')
parser.add_argument('-n', '--name', dest='name', required=False, help='Model name')
parser.add_argument('-m', '--model', dest='model', required=True,
                    help='swav/path_to_the_pre-trained_model.')
parser.add_argument('--lr', dest='max_lr', type=float, required=True, help='Maximal learning rate.')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                    required=False, default=100, help='Batch size.')
args = parser.parse_args()
dataset = args.data

with open(f'../data_splits/{dataset}-split.pkl', 'rb') as f:
    data_partition = pickle.load(f)
with open(f'../data_splits/{dataset}-le.pkl', 'rb') as f:
    le = pickle.load(f)
nr_labels = len(data_partition['train']['label'][0])
print(nr_labels)

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
      
    label = self.image_list['label'][idx].astype('float')

    return img, label

class Rot90:
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, [1, 2])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(292),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        Rot90(),
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

        state_dict = torch.load(os.path.join('../models/ssl', self.model_name + '.pth'))
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

def train_model(model, criterion, optimizer, scheduler, warmup, num_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    H = {'train': {'loss': [], 'metrics': []},
         'test': {'loss': [], 'metrics': []}}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            predictions = []
            ground_truth = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch)
                        warmup.dampen()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                predictions.append(torch.sigmoid(outputs).cpu().detach().numpy())
                ground_truth.append(labels.cpu().numpy())
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            predictions = np.concatenate(predictions, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            epoch_p, epoch_r, epoch_f1, _ = precision_recall_fscore_support(ground_truth, predictions>0.5, average='micro')

            print('{} Loss: {:.4f} Prec: {:.4f} Rec: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_p, epoch_r, epoch_f1))
            H[phase]['loss'].append(epoch_loss)
            H[phase]['metrics'].append((epoch_p, epoch_r, epoch_f1))

            # deep copy the model
            #if phase == 'test' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, H

image_datasets = {x: RSDataset(args.path, data_partition[x], data_transforms[x]) 
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
steps_per_epoch = dataset_sizes['train'] // args.batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

softmax = nn.Linear(2048, nr_labels)

if args.model == 'swav':
    backbone = ResNetBackbone(args.model)
    model_conv = nn.Sequential(backbone, softmax)
    if torch.cuda.device_count() > 1:
        model_conv = nn.DataParallel(model_conv)
else:
    model_conv = torch.load(args.model)
    model_conv.module[1] = softmax

for param in model_conv.module.parameters():
    param.requires_grad = True

model_conv = model_conv.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer_conv = optim.Adam(model_conv.parameters(), lr=args.max_lr)

scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=[50, 70, 90], gamma=0.2)
warmup_scheduler = warmup.LinearWarmup(optimizer_conv, 5*steps_per_epoch)

model_conv, hist = train_model(model_conv, criterion, optimizer_conv,
                               scheduler, warmup_scheduler, num_epochs=EPOCHS)

torch.save(model_conv, filename)
filename = '../models/rssc_resnet50'
if args.name is not None:
    filename += '_{}'.format(args.name)
filename += '_{}'.format(dataset)
filename += '_ft.pth'
torch.save(model_conv, filename)
