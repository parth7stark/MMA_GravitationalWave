import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
import pickle
import glob
# Add the directory containing data_generators_torch.py to the sys.path
sys.path.append('/scratch/bblq/parthpatel7173/APPFL/FedCompass/src/appfl/trainer/')
from noise_snr_schedule import *

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class whiten:
    @staticmethod
    def whiten(strain, interp_psd, dt):
        """
        Whitens the input strain signal.

        Parameters:
        - strain: Input strain signal.
        - interp_psd: Interpolated power spectral density (PSD) corresponding to the strain signal.
        - dt: Sampling interval of the strain signal.

        Returns:
        - white_ht: Whitened strain signal.
        """
        Nt = len(strain)
        freqs = np.fft.rfftfreq(Nt, dt)

        # whitening: transform to freq domain, divide by asd, then transform back,
        # taking care to get normalization right.

        # Transform to frequency domain
        hf = np.fft.rfft(strain)

        # Normalize to ensure correct scaling
        norm = 1./np.sqrt(1./(dt*2))

        # Divide Fourier transform of strain signal by square root of PSD
        white_hf = hf / np.sqrt(interp_psd(freqs)) * norm

        # Transform back to time domain
        white_ht = np.fft.irfft(white_hf, n=Nt)
        return white_ht

    @staticmethod
    def whiten_signal(strain_L1, strain_H1, strain_V1, dt, psd_L1, psd_H1, psd_V1):
        """
        Whitens three different strain signals from different detectors.

        Parameters:
        - strain_L1, strain_H1, strain_V1: Strain signals from detectors L1, H1, and V1.
        - dt: Sampling interval of the strain signals.
        - psd_L1, psd_H1, psd_V1: Power spectral densities (PSDs) corresponding to the strain signals.

        Returns:
        - Whitened strain signals from detectors L1, H1, and V1.
        """

        # Whitening each strain signal using respective PSD
        strain_whiten_L = whiten.whiten(strain_L1, psd_L1, dt)
        strain_whiten_H = whiten.whiten(strain_H1, psd_H1, dt)
        strain_whiten_V = whiten.whiten(strain_V1, psd_V1, dt)

        # Normalizing each whitened strain signal
        strain_whiten_L /= np.amax(np.absolute(strain_whiten_L))
        strain_whiten_H /= np.amax(np.absolute(strain_whiten_H))
        strain_whiten_V /= np.amax(np.absolute(strain_whiten_V))

        return strain_whiten_L, strain_whiten_H, strain_whiten_V

    @staticmethod
    def get_whitened_ligo_noise_chunk(strain, noise_strain_fn, gaussian=0):
        """
        Loads a chunk of LIGO noise data and extracts noise signals.

        Parameters:
        - strain: Input strain signal.
        - noise_strain_fn: Filename of the HDF5 file containing noise data.
        - gaussian: Flag indicating whether to use Gaussian noise.

        Returns:
        - Noise signals corresponding to detectors L1, H1, and V1.
        """

        f = h5py.File(noise_strain_fn, 'r')
        strain_L1 = f['strain_L1']
        strain_H1 = f['strain_H1']
        strain_V1 = f['strain_H1']
        if gaussian:
            strain_V1 = f['strain_V1']

        starting_index = np.random.randint(0, len(strain_H1)-len(strain))

        ligo_noise_L = np.zeros(len(strain))
        ligo_noise_H = np.zeros(len(strain))
        ligo_noise_V = np.zeros(len(strain))

        ligo_noise_L[:] = strain_L1[starting_index:starting_index+len(strain)]
        ligo_noise_H[:] = strain_H1[starting_index:starting_index+len(strain)]
        ligo_noise_V[:] = strain_V1[starting_index:starting_index+len(strain)]

        f.close()

        return ligo_noise_L, ligo_noise_H, ligo_noise_V
        
        

    @staticmethod
    def mix_signal_and_noise(strain_whiten_L, strain_whiten_H, strain_whiten_V, ligo_noise_L, ligo_noise_H, ligo_noise_V, noise_range):
        
        """
        Mixes whitened strain signals with whitened LIGO noise signals.

        Parameters:
        - strain_whiten_L, strain_whiten_H, strain_whiten_V: Whitened strain signals from detectors L1, H1, and V1.
        - ligo_noise_L, ligo_noise_H, ligo_noise_V: Whitened LIGO noise signals from detectors L1, H1, and V1.
        - noise_range: Range of target standard deviation for scaling noise signals.

        Returns:
        - Mixed signals from detectors L1, H1, and V1.
        """
        # Selecting a target standard deviation for scaling noise signals
        target_std = np.random.uniform(noise_range[0], noise_range[1])

        # Calculating standard deviations of noise signals
        ligo_noise_whiten_std_L = np.std(ligo_noise_L)
        ligo_noise_whiten_std_H = np.std(ligo_noise_H)
        ligo_noise_whiten_std_V = np.std(ligo_noise_V)

        # Scaling noise signals to match target standard deviation
        ligo_noise_whiten_L = target_std * (ligo_noise_L / ligo_noise_whiten_std_L)
        ligo_noise_whiten_H = target_std * (ligo_noise_H / ligo_noise_whiten_std_H)
        ligo_noise_whiten_V = target_std * (ligo_noise_V / ligo_noise_whiten_std_V)

        # Scaling noise signals to match target standard deviation
        mixed_L = strain_whiten_L + ligo_noise_whiten_L
        mixed_H = strain_whiten_H + ligo_noise_whiten_H
        mixed_V = strain_whiten_V + ligo_noise_whiten_V

        # Normalizing mixed signals
        mixed_std_L = np.std(mixed_L)
        mixed_std_H = np.std(mixed_H)
        mixed_std_V = np.std(mixed_V)

        mixed_L = mixed_L / mixed_std_L
        mixed_H = mixed_H / mixed_std_H
        mixed_V = mixed_V / mixed_std_V

        return mixed_L, mixed_H, mixed_V

    @staticmethod
    def single_shift(strain, merger, merger_shift, truncation):

        """
        Shifts a strain signal and its corresponding target labels to align with a merger time.

        Parameters:
        - strain: Input strain signal.
        - merger: Merger time.
        - merger_shift: Amount of shift.
        - truncation: Truncation value.

        Returns:
        - Shifted strain signal and target labels.
        """
        
        # Creating target signal around the merger time
        target = np.zeros(8192)
        target[max(truncation, merger-2048):merger+1] = 1

        if merger_shift <= merger:
            # Shifting strain signal and target labels
            strain = strain[merger-merger_shift:merger-merger_shift+4096]
            target = target[merger-merger_shift:merger-merger_shift+4096]
        else:
            tmp = strain[:4096-(merger_shift-merger)]
            tmp_target = target[:4096-(merger_shift-merger)]

            strain = np.zeros(4096)
            target = np.zeros(4096)

            strain[merger_shift-merger:] = tmp[:]
            target[merger_shift-merger:] = tmp_target[:]

        return strain, target


class WFGDataset(Dataset):
    def __init__(self, noise_dir, train_file, batch_size=32, dim=4096, n_channels=2,
                 shuffle=True, train=1, gaussian=0, noise_prob=0.6, noise_range=None):

        """
        Initializes the dataset with necessary parameters.

        Args:
        - noise_dir (str): Directory containing noise data.
        - data_dir (str): Directory containing waveform data.
        - train_file (str): File path for training or testing data.
        - batch_size (int): Batch size for data loading.
        - dim (int): Dimensionality of the waveform data.
        - n_channels (int): Number of channels in the data.
        - shuffle (bool): Whether to shuffle the dataset.
        - train (int): Flag indicating whether the dataset is for training (1) or testing (0).
        - gaussian (int): Flag indicating whether to use Gaussian noise.
        - noise_prob (float): Probability of adding noise to the data.
        - noise_range (tuple): Range for noise scaling.
        """

        # self.data_dir = data_dir
        self.wf_file = train_file
        self.noise_dir = noise_dir

        # # Determine the file paths based on the training flag
        # if train:
        #     self.wf_file = self.data_dir + 'train_300.hdf'
        # else:
        #     self.wf_file = self.data_dir + 'test_300.hdf'
        # row1 [amplitude] (0,0,1, 0.. 4096 samples)
        # Set up noise-related parameters based on the Gaussian flag
        self.gaussian = gaussian
        if gaussian:
            self.psd_L1_files = self.noise_dir + 'psd_L.pkl'
            self.psd_H1_files = self.noise_dir + 'psd_H.pkl'
            self.psd_V1_files = self.noise_dir + 'psd_V.pkl'
            self.noise_files = sorted(glob.glob(self.noise_dir + 'gaussian_4096_*'))
        else:
            self.psd_H1_files = sorted(glob.glob(self.noise_dir + '/processed_noise/psd_H*'))
            self.psd_L1_files = sorted(glob.glob(self.noise_dir + '/processed_noise/psd_L*'))
            self.noise_files = sorted(glob.glob(self.noise_dir + '/processed_noise/whitened_strains*'))

        # Open the waveform file and extract necessary keys
        self.f = h5py.File(self.wf_file, 'r')
        self.keys = list(self.f.keys())
        self.L1_wset = self.f[self.keys[0]+"/L1_wave"]
        self.H1_wset = self.f[self.keys[0]+"/H1_wave"]
        self.V1_wset = self.f[self.keys[0]+"/V1_wave"]

        self.fs = 4096
        self.dt = 1/self.fs
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.epoch = 1
        self.indices = np.arange(len(self.L1_wset))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.L1_wset)
        
    def increment_epoch(self):
        """
        Increments the epoch counter for training.
        """
        self.epoch += 1

    def __getitem__(self, index):
        """
        Fetches a sample and its corresponding label from the dataset.

        Args:
        - index (int): Index of the sample to retrieve.

        Returns:
        - X (torch.Tensor): Input data (waveform).
        - y (torch.Tensor): Target label.
        """
        idx = self.indices[index]
        X, y = self.__data_generation(idx)
        return torch.from_numpy(X), torch.from_numpy(y)

    def __data_generation(self, indexes):
        """
        Generates waveform data and labels for a single sample.

        Args:
        - indexes (int): Index of the sample to generate data for.

        Returns:
        - X (np.ndarray): Input data (waveform).
        - y (np.ndarray): Target label.
        """

        # Initialize arrays for data and labels
        X = np.zeros((self.dim, self.n_channels), dtype=np.float32)  
        y = np.zeros((self.dim, 1), dtype=np.float32)
        

        # Extract waveform data for each detector (L1, H1, V1)
        strains_L1 = self.L1_wset[indexes, 2048:-2048]
        strains_H1 = self.H1_wset[indexes, 2048:-2048]
        strains_V1 = self.V1_wset[indexes, 2048:-2048]
        

        '''
        Expand Dimensions:
        Expand the dimensions of the waveform data to match the batch size.
        This is necessary to concatenate the data along the batch dimension later.
        '''
        strains_L1 = np.expand_dims(strains_L1, axis=0)
        strains_H1 = np.expand_dims(strains_H1, axis=0)
        strains_V1 = np.expand_dims(strains_V1, axis=0)
        
        # Add noise to the data with a certain probability
        i=0
        if np.random.random_sample() > self.noise_prob:

            # Initialize arrays for noise signals
            strain_L1 = np.zeros(2*strains_L1.shape[1])
            strain_L1[:strains_L1.shape[1]] = strains_L1[i]

            strain_H1 = np.zeros(2*strains_H1.shape[1])
            strain_H1[:strains_H1.shape[1]] = strains_H1[i]

            strain_V1 = np.zeros(2*strains_V1.shape[1])
            strain_V1[:strains_V1.shape[1]] = strains_V1[i]


            # Find the position of the merger in each detector
            merger_L1 = np.argmax(np.absolute(strain_L1[:]))
            merger_H1 = np.argmax(np.absolute(strain_H1[:]))
            merger_V1 = np.argmax(np.absolute(strain_V1[:]))
            
            # Load PSD and noise files based on the Gaussian flag
            if self.gaussian:
                file_idx = np.random.randint(0, 3)
                psd_L1 = pickle.load(open(self.psd_L1_files, 'rb'), encoding="bytes")
                psd_H1 = pickle.load(open(self.psd_H1_files, 'rb'), encoding="bytes")
                psd_V1 = pickle.load(open(self.psd_V1_files, 'rb'), encoding="bytes")
                whitened_noise_strain_fn = self.noise_files[file_idx]
            else:
                file_idx = np.random.randint(0, 3)
                psd_L1 = pickle.load(open(self.psd_L1_files[file_idx], 'rb'), encoding="bytes")[0]
                psd_H1 = pickle.load(open(self.psd_H1_files[file_idx], 'rb'), encoding="bytes")[0]
                psd_V1 = pickle.load(open(self.psd_H1_files[file_idx], 'rb'), encoding="bytes")[0]
                whitened_noise_strain_fn = self.noise_files[file_idx]

            #  Generate and process noise signals
            ligo_noise_L, ligo_noise_H, ligo_noise_V = whiten.get_whitened_ligo_noise_chunk(strain_L1[:4096], whitened_noise_strain_fn,gaussian=self.gaussian)

            strain_whiten_L, strain_whiten_H, strain_whiten_V = whiten.whiten_signal(strain_L1, strain_H1, strain_V1, self.dt, psd_L1, psd_H1, psd_V1)

            #  Perform truncation and alignment
            #  truncation determines the length of the portion to be zeroed out
            truncation =  150
            strain_whiten_L[:truncation] = 0
            strain_whiten_L[-truncation:] = 0

            strain_whiten_H[:truncation] = 0
            strain_whiten_H[-truncation:] = 0

            strain_whiten_V[:truncation] = 0
            strain_whiten_V[-truncation:] = 0

            merger_shift = np.random.randint(4096//2, 4096)
            strain_whiten_L, target_L = whiten.single_shift(strain_whiten_L, merger_L1, merger_shift, truncation)
            strain_whiten_H, target_H = whiten.single_shift(strain_whiten_H, merger_H1, merger_H1+(merger_shift-merger_L1), truncation)
            strain_whiten_V, target_V = whiten.single_shift(strain_whiten_V, merger_V1, merger_V1+(merger_shift-merger_L1), truncation)

            # Determine noise range and mix signals with noise
            if not self.noise_range:
                noise_range = low_max_snr(self.epoch, noise_range_map)
            else:
                noise_range = self.noise_range
            mixed_L, mixed_H, mixed_V = whiten.mix_signal_and_noise(strain_whiten_L, strain_whiten_H, strain_whiten_V, \
                                                          ligo_noise_L, ligo_noise_H, ligo_noise_V, noise_range)


             # Assign data and labels                
            X[:, 0] = mixed_L
            X[:, 1] = mixed_H
            X[:, 2] = mixed_V

            y[:, 0] = target_H

        else:
            # If no noise is added, use only noise signals without any additional processing
            strain = np.zeros(strains_L1.shape[1])
            file_idx = np.random.randint(0, 3)
            whitened_noise_strain_fn = self.noise_files[file_idx]

            ligo_noise_L, ligo_noise_H, ligo_noise_V = whiten.get_whitened_ligo_noise_chunk(strain[:4096], whitened_noise_strain_fn,
                                                                                                gaussian=self.gaussian)


                
            X[:, 0] = (ligo_noise_L/np.std(ligo_noise_L))
            X[:, 1] = (ligo_noise_H/np.std(ligo_noise_H))
            X[:, 2] = (ligo_noise_V/np.std(ligo_noise_V))
            y[:, 0] = np.zeros(self.dim)

        return X, y
        
class WaveformDataModule(LightningDataModule):
    def __init__(self, noise_dir, train_file, val_file, batch_size=32, dim=4096, n_channels=3,
                 shuffle=True, gaussian=0, noise_prob=0.6, noise_range=None, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.dim=dim
        self.gaussian=gaussian
        self.shuffle=shuffle
        # self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.noise_dir = noise_dir
        self.n_channels = n_channels
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = WFGDataset(self.noise_dir, self.train_file, self.batch_size,self.dim, self.n_channels, True, 1,self.gaussian, self.noise_prob, self.noise_range)
            self.val_dataset = WFGDataset(self.noise_dir, self.val_file, self.batch_size,self.dim, self.n_channels, False, 0,self.gaussian, self.noise_prob, self.noise_range)

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=self.shuffle)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler,  num_workers=self.num_workers,pin_memory=True)

    def val_dataloader(self):
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers,pin_memory=True)
