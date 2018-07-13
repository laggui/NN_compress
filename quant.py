import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import OrderedDict

from VGG_base import VGG

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

def min_max_quantize(input, bits):
    # Unsigned min_max fixed-point quantization
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.numpy())
        min_val = float(min_val.data.numpy())
    
    input_rescale = (input - min_val) / (max_val - min_val)

    max_fixed = (2 ** bits) - 1
    result = torch.round(input_rescale * max_fixed) # Quantized tensor
    # This is a simulation. PyTorch currently doesn't support low-precision operations.
    # Operations would normally happen (here) with quantized tensor before dequantization.
    result =  (result / max_fixed) * (max_val - min_val) + min_val # Dequantize
    return result

def min_max_quantize_adjustedrange(input, bits):
    # Unsigned min_max fixed-point quantization with adjusted range to minimize error (compared to min_max_quantize)
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    min_val, max_val = input.min(), input.max()
    [min_fixed, max_fixed] = [0, (2 ** bits) - 1]

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.numpy())
        min_val = float(min_val.data.numpy())
    
    # Find adjusted range
    n = (2 ** bits)
    range_adjust = n / (n - 1)
    range = (max_val - min_val) * range_adjust
    
    # Compute scaling factor
    sf = n / range
    
    result = torch.round(input * sf) - round(min_val * sf) + min_fixed # Quantized tensor
    # This is a simulation. PyTorch currently doesn't support low-precision operations.
    # Operations would normally happen (here) with quantized tensor before dequantization.
    result =  (result - min_fixed + min_val * sf) / sf # Dequantize
    return result

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

def duplicate_model_with_quant(model, bits, type):
    """assume that original model has at least a nn.Sequential"""
    assert type is 'minmax'
    if isinstance(model, nn.Sequential):
        layer = OrderedDict()
        for key, val in model._modules.items():
            if isinstance(val, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)): #nn.Conv2d, nn.Linear, nn.BatchNorm2d
                layer[key] = val                
                quant_layer = NormalQuant('{}_quant'.format(key), bits=bits, quant_func=min_max_quantize)
                layer['{}_{}_quant'.format(key, type)] = quant_layer
            else:
                layer[key] = duplicate_model_with_quant(val, bits, type)
        m = nn.Sequential(layer)
        return m
    else:
        for key, val in model._modules.items():
            model._modules[key] = duplicate_model_with_quant(val, bits, type)
        return model

# Check device    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # Ran out of memory on GPU
# Model
print('==> Building model...')
model = VGG('D', num_classes=10, input_size=32) # VGG16 is configuration D (refer to paper)
# Load model
print('==> Loading model...')
model.load_state_dict(torch.load('VGG16model_timed.pth'))

state_dict = model.state_dict()
NUM_BITS = 8
model_quant = model

# Quantize parameters
state_dict_quant = OrderedDict()
for key, val in state_dict.items():
    v_quant = min_max_quantize_adjustedrange(val, NUM_BITS)
    state_dict_quant[key] = v_quant
    #print(key, NUM_BITS)
model_quant.load_state_dict(state_dict_quant)

# Quantize activations
model_quant = duplicate_model_with_quant(model_quant, NUM_BITS, type='minmax')

print('==> Finished quantization')



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if device == 'cuda':
    model_quant = model_quant.cuda()
    cudnn.benchmark = True
    model_quant = torch.nn.DataParallel(model_quant)

model_quant = model_quant.cpu() # Ran out of memory otherwise when trying to test on GPU

correct, total = test(model_quant, test_loader)
print('Accuracy of the network on test dataset: %f (%d/%d)' % (100.*correct/total, correct, total))