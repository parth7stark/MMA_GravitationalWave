'''
Below is the Python code that defines the client-side model architecture. The client-side model will handle the HDCN (Hierarchical Dilated Convolutional Network) block
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubModule(nn.Module):
    def __init__(self, inp_shape=(4096, 1)):
        super(SubModule, self).__init__()

        self.n_filters = 32
        self.filter_width = 2
        self.dilation_rates = [2**i for i in range(11)] * 3


        self.conv1_firstit = nn.Conv1d(1,16, kernel_size=1,  padding='same')
        self.conv1_postfirstit = nn.ModuleList([nn.Conv1d(16,16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])[:-1]
        self.convs_f = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.convs_g = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.conv2 = nn.ModuleList([nn.Conv1d(self.n_filters, 16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])

    def forward(self, x):
        skips = []
        for i, dilation_rate in enumerate(self.dilation_rates):
            
            conv1=self.conv1_firstit if i==0 else self.conv1_postfirstit[i-1]
            
            x = F.relu(conv1(x))
            x_f = self.convs_f[i](x)
            x_g = self.convs_g[i](x)
            z = F.tanh(x_f) * F.sigmoid(x_g)
            z = F.relu(self.conv2[i](z))
            x = x + z
            skips.append(z)
        out = F.relu(torch.sum(torch.stack(skips), dim=0))
        return out

class ClientModel(nn.Module):
    '''
    ClientModel: The ClientModel uses the SubModule class to generate embeddings from the input time series data. This is the HDCN block that each detector site (client) will use to process its local data and generate embeddings.
    '''
    def __init__(self):
        super(ClientModel, self).__init__()
        self.sub_mod = SubModule()

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # Change the shape to (batch_size, channels, length)

        x_A = x[:, :, 0].view(-1, 4096, 1).permute(0,2,1) 
        embeddings = self.sub_mod(x_A)
        return embeddings
