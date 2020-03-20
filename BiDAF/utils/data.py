import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TweetData(Dataset):
    def __init__(self, file_path):
        self.data_frame = json.load(open(file_path))
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tweet = torch.tensor(self.data_frame[idx]['Tweet'], dtype=torch.float)
        question = torch.tensor(self.data_frame[idx]['Question'], dtype=torch.float)
        ans = torch.tensor(self.data_frame[idx]['Answer'][0], dtype=torch.long)
        return tweet, question, ans