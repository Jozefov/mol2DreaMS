from torch.utils.data import Dataset
from torch_geometric.data import Batch

class SimpleDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)