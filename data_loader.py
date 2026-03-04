import torch
from torch.utils.data import Dataset
import numpy as np
from utils import extract_sequence, one_hot_encode

class GeneExpressionDataset(Dataset):
    def __init__(self, gene_df, expr_df, fasta, window):
        self.gene_df = gene_df
        self.expr_df = expr_df
        self.fasta = fasta
        self.window = window
        self.indices = []
        self.seqs = []
        self.targets = []

        for idx, row in gene_df.iterrows():
            gene_id = row['gene_id']
            if gene_id not in expr_df.index:
                continue
            seq = extract_sequence(row, fasta, window)
            if seq is None:
                continue
            if len(seq) != 2*window + 1:
                continue
            target = expr_df.loc[gene_id].values.astype(np.float32)
            self.indices.append(idx)
            self.seqs.append(seq)
            self.targets.append(target)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        target = self.targets[idx]
        X = one_hot_encode(seq)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return X, y
