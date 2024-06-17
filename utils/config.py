
class Config:
    def __init__(self):
        # Model parameters
        self.vocab_size = 21  # 20 amino acids + 1 padding index
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.num_layers = 6
        self.dropout = 0.1

        # Training parameters
        self.batch_size = 64
        self.num_epochs = 10
        self.learning_rate = 0.0001
        self.max_seq_len = 100

        # Special tokens
        self.pad_idx = 20
        self.start_idx = 21  # Add a start token if needed
        self.end_idx = 22    # Add an end token if needed

        # Amino acid vocabulary
        self.amino_acid_vocab = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }

    def __str__(self):
        return ', '.join(f'{k}={v}' for k, v in self.__dict__.items())