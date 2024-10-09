from torch.utils.data import Dataset
from torch_geometric.data import Batch
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, triplets_df, featurized_molecules):
        self.featurized_molecules = featurized_molecules
        self.valid_identifiers = set(featurized_molecules.keys())

        # Pre-filter triplets to ensure all molecules are featurized and have valid positive and negative IDs
        valid_triplets = []
        for idx, row in triplets_df.iterrows():
            anchor_id = row['anchor_id']

            if anchor_id not in self.valid_identifiers:
                continue

            # Filter positive and negative IDs to only those that are featurized
            positive_ids = [pid for pid in row['positive_ids'] if pid in self.valid_identifiers]
            negative_ids = [nid for nid in row['negative_ids'] if nid in self.valid_identifiers]

            if not positive_ids or not negative_ids:
                continue

            valid_triplets.append({
                'anchor_id': anchor_id,
                'positive_ids': positive_ids,
                'negative_ids': negative_ids
            })

        self.triplets = valid_triplets
        print(f"Total valid triplets: {len(self.triplets)}")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor_id = triplet['anchor_id']
        positive_ids = triplet['positive_ids']
        negative_ids = triplet['negative_ids']

        # Randomly select one positive and one negative example
        positive_id = np.random.choice(positive_ids)
        negative_id = np.random.choice(negative_ids)

        anchor_data = self.featurized_molecules[anchor_id]
        positive_data = self.featurized_molecules[positive_id]
        negative_data = self.featurized_molecules[negative_id]

        return anchor_data, positive_data, negative_data

    @staticmethod
    def collate_fn(batch):
        anchor_list, positive_list, negative_list = zip(*batch)

        # Combine the lists into batches
        anchor_batch = Batch.from_data_list(anchor_list)
        positive_batch = Batch.from_data_list(positive_list)
        negative_batch = Batch.from_data_list(negative_list)

        return anchor_batch, positive_batch, negative_batch