import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from VGG_base import VGG

# Check device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader):
    model.eval()
    print_freq = 10 # print every 10 batches
    correct = 0
    total = 0
    
    with torch.no_grad(): # no need to track history
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)        

            # record prediction accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % print_freq == 0:
                print('Batch: %d, Acc: %.3f%% (%d/%d)' % (batch_idx+1, 100.*correct/total, correct, total))
    return correct, total

# Model
print('==> Building model...')
model = VGG('D', num_classes=10, input_size=32) # VGG16 is configuration D (refer to paper)
# Load model
print('==> Loading model...')
model.load_state_dict(torch.load('VGG16model.pth'))
state_dict = model.state_dict()
model = model.to(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if device == 'cuda':
	cudnn.benchmark = True
	model = torch.nn.DataParallel(model)

correct, total = test(model, test_loader)
print('Accuracy of the network on test dataset: %f (%d/%d)' % (100.*correct/total, correct, total))