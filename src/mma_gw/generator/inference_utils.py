import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# def normalize(strain):
#     std = np.std(strain[:])
#     strain[:] /= std
#     return strain

def normalize(strain):
    std = np.std(strain[:])
    mean=np.mean(strain[:])
    strain[:]+=-mean
    strain[:] /= std
    return strain

def preprocess(strain_data, length):
    normalized_strain_data = normalize(strain_data)
    # acc to me - check with victoria
    # length is used for testing the code --> only take a certain length of strain for inference
    # minyang doesn't have this part
    if length != "None":
        normalized_strain_data = torch.tensor(normalized_strain_data[:length])
    else:
        normalized_strain_data = torch.tensor(normalized_strain_data)
        
    # data = torch.stack((normalized_strain_data), dim=1)
    data = normalized_strain_data.unsqueeze(1)  # Add dimension for single detector
    print("data shape:", data.shape)
    return data

#custom sampler        
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, length, stride=1, start_index=0, end_index=None):
        self.data = data
        self.targets = targets
        self.length = length
        self.stride = stride
        
        if end_index is None or end_index > len(data):
            end_index = len(data) - 1

        self.start_index = start_index
        self.end_index = end_index
        
        self.sample_indices = np.arange(self.start_index, self.end_index - self.length + 1, self.stride)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]
        end_idx = start_idx + self.length

        # Slice the data and targets for the current sample
        sample_data = self.data[start_idx:end_idx]
        sample_target = self.targets[start_idx:end_idx]

        return sample_data.float(), sample_target.float()

