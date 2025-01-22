'''

Below is the Python code that defines the server-side model architecture. The server-side model will handle the GNN (Graph Neural Network) block.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PinSage(nn.Module):
    def __init__(self, dim, node_dim):
        super(PinSage, self).__init__()
        self.dim = dim
        self.node_dim=node_dim
        self.neighbor_aggregation1 = nn.Conv1d(16,dim, kernel_size=1)
        self.update_target_node = nn.Conv1d(32,self.node_dim, kernel_size=1)

    def forward(self, target_node, neighbor_1):
        neighbor_1 = F.relu(self.neighbor_aggregation1(neighbor_1)).permute(0,2,1).view(-1, 4096, self.dim, 1)
        neighbors = neighbor_1.squeeze(-1)


        out_node = torch.cat([target_node.permute(0,2,1), neighbors], dim=-1)
        out_node = F.relu(self.update_target_node(out_node.permute(0,2,1)))
        return out_node

class ServerModel(nn.Module):
    def __init__(self, node_dim=64):
        super(ServerModel, self).__init__()
        self.dim = 16
        self.node_dim = node_dim

        self.pinsage_A = PinSage(self.dim, self.node_dim)
        self.pinsage_B = PinSage(self.dim, self.node_dim)
        self.conv1d = nn.Conv1d(self.node_dim, 1, 1)

    def forward(self, x_A, x_B):

        # Aggregate information using PinSage layers
        updated_x_A = self.pinsage_A(x_A, x_B).permute(0, 2, 1).view(-1, 4096, self.node_dim, 1)
        updated_x_B = self.pinsage_B(x_B, x_A).permute(0, 2, 1).view(-1, 4096, self.node_dim, 1)

        # Concatenate updated node representations
        out = torch.cat([updated_x_A, updated_x_B], dim=-1)

        # Apply max pooling
        out = torch.max(out, dim=-1).values

        # Apply final convolutional layer
        out = self.conv1d(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = F.sigmoid(out)

        return out
