import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from google.colab import files
src = list(files.upload().values())[0]
open('VGG_base.py','wb').write(src)
from VGG_base import VGG

# Check device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    print_freq = 10 # print every 10 batches
    train_loss = 0
    correct = 0
    total = 0
    print('\nEpoch: %d' % epoch)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # record loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % print_freq == 0:
            print('Batch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def validate(model, val_loader, criterion):
    model.eval()
    print_freq = 10 # print every 10 batches
    val_loss = 0.0
    
    with torch.no_grad(): # no need to track history
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)        
            loss = criterion(outputs, targets)

            # record loss
            val_loss += loss.item()

            if batch_idx % print_freq == 0:
                print('Validation on Batch: %d, Loss: %f' % (batch_idx+1, val_loss/(batch_idx+1)))
    return val_loss

# Load CIFAR10 dataset
print('==> Preparing data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Model
print('==> Building model...')
model = VGG('D', num_classes=10, input_size=32) # VGG16 is configuration D (refer to paper)
model = model.to(device)

if device == 'cuda':
	cudnn.benchmark = True
	model = torch.nn.DataParallel(model)

# Training
num_epochs = 200 # as opposed to the paper (74) because of CIFAR10 dataset
lr = 0.1
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

print('==> Training...')
#scheduler = ReduceLROnPlateau(optimizer, 'min')
scheduler = StepLR(optimizer, step_size=100, gamma=0.1) # adjust lr by factor of 10 every 100 epochs
for epoch in range(num_epochs):
    # train one epoch
    train(model, train_loader, criterion, optimizer, epoch)
    # validate
    #val_loss = validate(model, val_loader, criterion)
    # adjust learning rate with scheduler
    #scheduler.step(val_loss)
    scheduler.step()
    
print('==> Finished Training')
# Save trained model
torch.save(model.state_dict(), 'VGG16model.pth')