# # Base VGG model implementation based on the popular paper https://arxiv.org/pdf/1409.1556.pdf
# 
# Here, configuration D is chosen (16-layer model, also known as VGG16):
# 64, 64, MaxPool, 128, 128, MaxPool, 256, 256, 256, MaxPool, 512, 512, 512, MaxPool, 512, 512, 512, MaxPool, FC
# But the implementation allows for other configurations to be added in the future (see cfg in _make_layers)
# 
# -------------------------
# Calculating the number of parameters for the fully connected layers below.
# 
# Depth: number of filters from conv
# 
# Pooling: the window is 2 x 2 and the stride is 2, the layer is outputting a pixel for every 2 x 2 pixels
# and jumping by 2 pixels to do the next calculation (no overlap), so the spatial resolution is divided by 
# 2 in each pooling layer
# 
# Thus, by starting with RGB images of size 224x224 pixels (as the paper specifiesd), with 5 MaxPool layers we get: 
# 224/(2^5) = 7 
# Input of first fully connected layer = output of last conv layer after MaxPool = 7x7x512
# 
# -------------------------
# Notes on weight initialization:
# 
# xavier_uniform()
# Fills the input Tensor with values according to the method described in Glorot, X. & Bengio, Y. (2010) 
# using a uniform distribution. 
# 
# From the paper: "It is worth noting that after the paper submission we found that it is possible to initialise 
# the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010)"
# "For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean
# and 10^−2 variance."


import torch
import torch.nn as nn

class VGG(nn.Module):
    '''
    Base VGG model
    '''
    def __init__(self, vgg_cfg,  num_classes=1000, input_size=224):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg)
        self.classifier = nn.Sequential(
                        nn.Linear(int((input_size/(2**5))*(input_size/(2**5))*512), 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(), # Dropout of 0.5 is default, as in paper
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                        )
        self._init_weights()

    def forward(self, x): # computation performed at every call
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        return x
    
    def _make_layers(self, vgg_cfg):
        '''
        For portability, if other configurations of the network 
        need to be defined (e.g., A, B, C or E from the VGG paper)
        ''' 
        cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
        
        in_channels = 3 # RGB images
        layers = []
        
        for conv in cfg[vgg_cfg]:
            if conv == 'M': # MaxPool: 2x2 kernel, 2px stride
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # Convolution: 3x3 filters, 1px stride, 1px padding            
                layers += [nn.Conv2d(in_channels, conv, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = conv # Next conv input is the size of current output
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules(): # Loop over each layer to initialize weights
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
