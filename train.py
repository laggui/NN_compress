import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from VGG_base import VGG

# Check device    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

if device == 'cuda:0':
	cudnn.benchmark = True

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    print_freq = 10 # print every 10 batches
    train_loss = 0
    print('Epoch:%d', epoch)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # compute output
        outputs = model(inputs)        
        loss = criterion(outputs, targets)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # record loss
        train_loss += loss.item()
        
        if batch_idx % print_freq == 0:
            print('Batch: %d, Loss: %f' % (batch_idx+1, (train_loss/(batch_idx+1))))

def validate(model, val_loader, criterion):
    model.eval()
    print_freq = 10 # print every 10 batches
    val_loss = 0.0
    batch_loss = 0.0
    
    with torch.no_grad(): # no need to track history
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)        
            loss = criterion(outputs, targets)

            # record loss
            batch_loss += loss.item()
            val_loss += loss.item()

            if batch_idx % print_freq == 0:
                print('Validation on Batch: %d, Loss: %f' % (batch_idx+1, batch_loss/print_freq))
                batch_loss = 0.0
    return val_loss

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment=False,
                           random_seed=3,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    '''
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Taken from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    '''
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

# Load CIFAR10 dataset
print('==> Preparing data...')

train_loader, val_loader = get_train_valid_loader(data_dir='./data', 
                                                  batch_size=128,
                                                  augment=True,
                                                  valid_size=0,												  
                                                  num_workers=2, 
                                                  pin_memory=True)

# Model
print('==> Building model...')
model = VGG('D', num_classes=10, input_size=32) # VGG16 is configuration D (refer to paper)
model = model.to(device)

# Training
num_epochs = 200 # as opposed to the paper (74) because of CIFAR10 dataset
lr = 0.1
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

print('==> Training...')
#scheduler = ReduceLROnPlateau(optimizer, 'min')
scheduler = StepLR(optimizer, step_size=50, gamma=0.1) # adjust lr by factor of 10 every 50 epochs
for epoch in range(num_epochs):
    # train one epoch
    train(model, train_loader, criterion, optimizer, epoch)
    # validate
    #val_loss = validate(model, val_loader, criterion)
    # adjust learning rate with scheduler
    #scheduler.step(val_loss)
    scheduler.step()
    
print('==> Finished Training')