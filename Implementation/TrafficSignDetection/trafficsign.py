# -*- coding: utf-8 -*-
"""trafficSign.ipynb


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pickle
import cv2
from torch.utils.data import Dataset, DataLoader
from matplotlib.pyplot import *

torch.cuda.set_device(0) 
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import pickle

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print ('Data loaded')

def reshape_raw_images(imgs):
    """Given 4D images (number, heigh, weight, channel), this
    function grayscales and returns (number, height, weight, 1) images"""
    def gray(src):
        if src.dtype == np.uint8:
            src = np.array(src/255.0, dtype=np.float32)
        dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        return dst.reshape(32,32,1)
    norms = [gray(img) for img in imgs]
    return np.array(norms)
    
features_train = reshape_raw_images(X_train)
labels_train   = torch.tensor(y_train, dtype=torch.long)
features_train = np.transpose(features_train, (0,3, 1, 2))


features_valid = reshape_raw_images(X_valid)
features_valid = np.transpose(features_valid,(0,3,1,2))
labels_valid   = torch.tensor(y_valid, dtype=torch.long)

features_test  = reshape_raw_images(X_test)
features_test = np.transpose(features_test,(0,3,1,2))
labels_test    = y_test

class train_dataset():
    """ Train Dataset loader"""

    # Initialize your data, download, etc.
    def __init__(self):
        self.len = features_train.shape[0]
        self.x_data = torch.from_numpy(features_train).cuda()
        self.y_data = labels_train.cuda() 

    def __getitem__(self, index):
        return self.x_data[index].cuda(), self.y_data[index].cuda() 

    def __len__(self):
        return self.len

    
class valid_dataset():
    """ Test Dataset loader"""

    # Initialize your data, download, etc.
    def __init__(self):
        self.len = features_valid.shape[0]
        self.x_data = torch.from_numpy(features_valid).cuda() 
        self.y_data = labels_valid.cuda() 

    def __getitem__(self, index):
        return self.x_data[index].cuda(), self.y_data[index].cuda() 

    def __len__(self):
        return self.len

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 1)       
        
        self.conv2 = nn.Conv2d(32, 64, (5,5))
                
        self.conv3 = nn.Conv2d(64, 128, (5,5))
        
        # input dim = 16*5*5, output dim = 120
        
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        
        # input dim = 120, output dim = 84
        
        self.fc2 = nn.Linear(1024, 512)
        
        # input dim = 84, output dim = 10
        
        self.fc3 = nn.Linear(512, 43)

    def forward(self, x):
        # pool size = 2
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        
        x = F.max_pool2d(x, 2)
        
#         x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # flatten as one dimension
        x = x.view(x.size()[0], -1)
        # input dim = 16*5*5, output dim = 120
        x = F.relu(self.fc1(x))
        # input dim = 120, output dim = 84
        x = F.relu(self.fc2(x))
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x

def load_data(train_batch_size, test_batch_size):
    # Fetch test data: total 34799 samples
    train_loader=DataLoader(dataset=train_dataset(),
                          batch_size=train_batch_size,
                          shuffle=True)
    
    # Fetch test data: total 10000 samples
    test_loader = DataLoader(dataset=valid_dataset(),
                            batch_size=test_batch_size,
                             shuffle=True)

    return (train_loader, test_loader)

def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the `Variable` wrapper
#         print(data.shape)
#         print(type(target))
        data, target = Variable(data), Variable(target)

        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        loss = loss_fn(output, target)

        # Backward propagation
        loss.backward()

        # Update the parameters(weight,bias)
        optimizer.step()

        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data))

def test(model, epoch, test_loader):
    # State that you are testing the model; this prevents layers e.g. Dropout to take effect
    model.eval()

    # Init loss & correct prediction accumulators
    test_loss = 0
    correct = 0

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

    # Iterate over data
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        
        # Forward propagation
        output = model(data)

        # Calculate & accumulate loss
        test_loss += loss_fn(output, target).data

        # Get the index of the max log-probability (the predicted output label)
        pred = np.argmax(output.data.cpu(), axis=1)

        # If correct, increment correct prediction accumulator
        correct = correct + np.equal(pred.cpu(), target.data.cpu()).sum()

    # Print log
    test_loss /= len(test_loader.dataset)
    print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

torch.manual_seed(123)

model = LeNet()

lr = 0.001
momentum=0.5
optimizer = optim.Adam(model.parameters(), lr=lr)

train_batch_size = 32
test_batch_size = 1000
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

epochs = 13
log_interval = 100
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)
    test(model, epoch, test_loader)
