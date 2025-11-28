import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle 
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )

def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData
        
def pad_to_multiple(tensor, multiple, dim=1, value=0):
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)

