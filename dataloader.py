from torch.utils.data import DataLoader
import torch

def get_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def collate_fn(batch, pad_token_id=0, max_len=100):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    # Padding sequences manually
    padded_input_ids = [
        torch.tensor(seq.tolist() + [pad_token_id] * (max_len - len(seq)))
        for seq in input_ids
    ]
    padded_input_ids = torch.stack(padded_input_ids)

    # Convert labels to tensor
    labels = torch.tensor(labels)

    return {
        'input_ids': padded_input_ids,
        'labels': labels
    }
 
