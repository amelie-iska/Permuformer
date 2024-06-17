
import torch
from torch.utils.data import Dataset
from utils.config import Config

class ProteinSequenceDataset(Dataset):
    def __init__(self, data_file, config):
        self.config = config
        self.data = self.load_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        sequence = self.preprocess(sequence)
        return sequence

    def load_data(self, data_file):
        sequences = []
        with open(data_file, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
        return sequences

    def preprocess(self, sequence):
        # Convert amino acid characters to indices
        indices = [self.config.amino_acid_vocab[aa] for aa in sequence]
        
        # Pad the sequence to a fixed length
        padding = [self.config.pad_idx] * (self.config.max_seq_len - len(indices))
        indices += padding
        indices = indices[:self.config.max_seq_len]
        
        return torch.tensor(indices, dtype=torch.long)