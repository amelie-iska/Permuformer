
from torch.utils.data import DataLoader
from data.dataset import ProteinSequenceDataset

def get_dataloader(config, data_file, batch_size=None, shuffle=True):
    dataset = ProteinSequenceDataset(data_file, config)
    batch_size = batch_size if batch_size is not None else config.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader