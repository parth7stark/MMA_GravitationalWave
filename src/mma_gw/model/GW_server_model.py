'''

Below is the Python code that defines the server-side model architecture. The server-side model will handle the GNN (Graph Neural Network) block.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PinSage_Attn(nn.Module):
    def __init__(self, dummy, dummy1, convolution_dim=16, num_heads=1):
        super().__init__()
        assert convolution_dim % num_heads == 0, \
            f"convolution_dim ({convolution_dim}) must be divisible by num_heads ({num_heads})."

        self.convolution_dim = convolution_dim
        self.num_heads = num_heads
        self.dim = convolution_dim // num_heads  # Dimension per head

        # Q, K, V Projections
        self.query_proj = nn.Linear(convolution_dim, convolution_dim)
        self.key_proj = nn.Linear(convolution_dim, convolution_dim)
        self.value_proj = nn.Linear(convolution_dim, convolution_dim)

        # Final projection
        self.output_proj = nn.Linear(convolution_dim, convolution_dim)

    def forward(self, target_node, neighbor):
        # Input shapes: (batch_size, convolution_dim, seq_len)
        batch_size, conv_dim, seq_len = target_node.size()

        # Debug shapes
        #print(f"Input shapes: target_node {target_node.shape}, neighbor {neighbor.shape}")

        # Permute for projection
        target_node = target_node.permute(0, 2, 1)  # (batch_size, seq_len, convolution_dim)
        neighbor = neighbor.permute(0, 2, 1)       # (batch_size, seq_len, convolution_dim)

        # Q, K, V Projections
        Q = self.query_proj(target_node)  # (batch_size, seq_len, convolution_dim)
        K = self.key_proj(neighbor)       # (batch_size, seq_len, convolution_dim)
        V = self.value_proj(neighbor)     # (batch_size, seq_len, convolution_dim)

        # Debug shapes
        #print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)

        # Debug shapes
        #print(f"Q reshaped: {Q.shape}, K reshaped: {K.shape}, V reshaped: {V.shape}")

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # Normalize scores
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, dim)

        # Debug shapes
        #print(f"Attention weights shape: {attention_weights.shape}, attended_values shape: {attended_values.shape}")

        # Concatenate heads and project output
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, dim)
        attended_values = attended_values.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, convolution_dim)
        updated_node = self.output_proj(attended_values)  # (batch_size, seq_len, convolution_dim)

        # Permute back
        output = updated_node.permute(0, 2, 1)  # (batch_size, convolution_dim, seq_len)

        # Debug final shape
        #print(f"Output shape: {output.shape}")
        return output


class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.dim = 16
        self.node_dim = 64

        self.pinsage_A = PinSage_Attn(self.dim, self.node_dim)
        self.pinsage_B = PinSage_Attn(self.dim, self.node_dim)        
        self.conv1d=nn.Conv1d(self.dim,1, 1) #change to dim for attn, node_dim for orig

    def forward(self, x_A, x_B):

        # Aggregate information using PinSage layers
        updated_x_A = self.pinsage_A(x_A, x_B).permute(0, 2, 1).view(-1, 4096, self.dim, 1) #change to dim for attn, node_dim for orig
        updated_x_B = self.pinsage_B(x_B, x_A).permute(0, 2, 1).view(-1, 4096, self.dim, 1) #change to dim for attn, node_dim for orig

        # Concatenate updated node representations
        out = torch.cat([updated_x_A, updated_x_B], dim=-1)

        # Apply max pooling
        out = torch.max(out, dim=-1).values

        # Apply final convolutional layer
        out = self.conv1d(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = F.sigmoid(out)

        return out

'''
Old Architecture: without self attention
'''
# class PinSage(nn.Module):
#     def __init__(self, dim, node_dim):
#         super(PinSage, self).__init__()
#         self.dim = dim
#         self.node_dim=node_dim
#         self.neighbor_aggregation1 = nn.Conv1d(16,dim, kernel_size=1)
#         self.update_target_node = nn.Conv1d(32,self.node_dim, kernel_size=1)

#     def forward(self, target_node, neighbor_1):
#         neighbor_1 = F.relu(self.neighbor_aggregation1(neighbor_1)).permute(0,2,1).view(-1, 4096, self.dim, 1)
#         neighbors = neighbor_1.squeeze(-1)


#         out_node = torch.cat([target_node.permute(0,2,1), neighbors], dim=-1)
#         out_node = F.relu(self.update_target_node(out_node.permute(0,2,1)))
#         return out_node

# class ServerModel(nn.Module):
#     def __init__(self, node_dim=64):
#         super(ServerModel, self).__init__()
#         self.dim = 16
#         self.node_dim = node_dim

#         self.pinsage_A = PinSage(self.dim, self.node_dim)
#         self.pinsage_B = PinSage(self.dim, self.node_dim)
#         self.conv1d = nn.Conv1d(self.node_dim, 1, 1)

#     def forward(self, x_A, x_B):

#         # Aggregate information using PinSage layers
#         updated_x_A = self.pinsage_A(x_A, x_B).permute(0, 2, 1).view(-1, 4096, self.node_dim, 1)
#         updated_x_B = self.pinsage_B(x_B, x_A).permute(0, 2, 1).view(-1, 4096, self.node_dim, 1)

#         # Concatenate updated node representations
#         out = torch.cat([updated_x_A, updated_x_B], dim=-1)

#         # Apply max pooling
#         out = torch.max(out, dim=-1).values

#         # Apply final convolutional layer
#         out = self.conv1d(out.permute(0, 2, 1)).permute(0, 2, 1)
#         out = F.sigmoid(out)

#         return out
