import torch
from torch.utils.data import Dataset
import h5py

class NHCPPDataset(Dataset):
    """  NHCPP dataset: download data from https://doi.org/10.5281/zenodo.5544042 """

    def __init__(self, file_path: str, mvts_length:int, normalize_mvts=True):
        """
        Args:
            file_path: path to the hdf5 data file
            mvts_length: length of the input sequence to use
            normalize_mvts: bool indicating if time-series should be normalized
        """
        self.__file_path = file_path
        self.__normalize = normalize_mvts
        self.mvts_length = mvts_length

        with h5py.File(file_path, 'r') as h5_file:
            attrs = dict(h5_file.attrs.items())
        self.__dataset_length = attrs['dataset_len']
        self.num_channels = attrs['num_channels']
        self.channels = attrs['channels']
        self.num_classes = attrs['num_classes']

    def __len__(self):
        return self.__dataset_length

    def __getitem__(self, idx):
        # read the hdf5 file and select the data to load in by index
        with h5py.File(self.__file_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            dset = h5_file[keys[idx]]
            mvts = dset[()]
            label = dset.attrs['degradation_state'].tolist()

        mvts = torch.tensor(mvts).squeeze().float()[:self.mvts_length, :].permute(1, 0)

        # per channel normalization
        if self.__normalize:
            mean = mvts.mean(dim=1).unsqueeze(1)
            std =  mvts.std(dim=1).unsqueeze(1)
            mvts = (mvts - mean) / std

        label = torch.tensor(label, dtype=torch.long).squeeze()
        return mvts, label